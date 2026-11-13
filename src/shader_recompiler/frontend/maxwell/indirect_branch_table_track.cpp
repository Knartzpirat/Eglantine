// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <optional>
#include <fmt/format.h>

#include "common/common_types.h"
#include "common/logging.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/maxwell/decode.h"
#include "shader_recompiler/frontend/maxwell/indirect_branch_table_track.h"
#include "shader_recompiler/frontend/maxwell/opcodes.h"
#include "shader_recompiler/frontend/maxwell/translate/impl/load_constant.h"

namespace Shader::Maxwell {
namespace {
union Encoding {
    u64 raw;
    BitField<0, 8, IR::Reg> dest_reg;
    BitField<8, 8, IR::Reg> src_reg;
    BitField<20, 19, u64> immediate;
    BitField<56, 1, u64> is_negative;
    BitField<20, 24, s64> brx_offset;
};

template <typename Callable>
std::optional<u64> Track(Environment& env, Location block_begin, Location& pos, Callable&& func) {
    while (pos >= block_begin) {
        const u64 insn{env.ReadInstruction(pos.Offset())};
        --pos;
        if (func(insn, Decode(insn))) {
            return insn;
        }
    }
    return std::nullopt;
}

std::optional<u64> TrackLDC(Environment& env, Location block_begin, Location& pos,
                            IR::Reg brx_reg) {
    return Track(env, block_begin, pos, [brx_reg](u64 insn, Opcode opcode) {
        const LDC::Encoding ldc{insn};
        return opcode == Opcode::LDC && ldc.dest_reg == brx_reg && ldc.size == LDC::Size::B32 &&
               ldc.mode == LDC::Mode::Default;
    });
}

std::optional<u64> TrackSHL(Environment& env, Location block_begin, Location& pos,
                            IR::Reg ldc_reg) {
    return Track(env, block_begin, pos, [ldc_reg](u64 insn, Opcode opcode) {
        const Encoding shl{insn};
        return opcode == Opcode::SHL_imm && shl.dest_reg == ldc_reg;
    });
}

std::optional<u64> TrackIMNMX(Environment& env, Location block_begin, Location& pos,
                              IR::Reg shl_reg) {
    return Track(env, block_begin, pos, [shl_reg](u64 insn, Opcode opcode) {
        const Encoding imnmx{insn};
        return opcode == Opcode::IMNMX_imm && imnmx.dest_reg == shl_reg;
    });
}

// Helper to skip MOV instructions and track register chains
IR::Reg SkipMovInstructions(Environment& env, Location block_begin, Location& pos,
                            IR::Reg target_reg) {
    IR::Reg current_reg{target_reg};

    while (pos >= block_begin) {
        const u64 next_insn{env.ReadInstruction(pos.Offset())};
        const Opcode next_opcode{Decode(next_insn)};

        if (next_opcode == Opcode::MOV_reg) {
            const Encoding mov{next_insn};
            if (mov.dest_reg == current_reg) {
                current_reg = mov.src_reg;
                --pos;
                continue;
            }
        }
        break;
    }

    return current_reg;
}

// Helper to skip intermediate arithmetic instructions (IADD, SHL, etc.)
// and track register transformations
IR::Reg SkipArithmetic(Environment& env, Location block_begin, Location& pos,
                       IR::Reg target_reg) {
    IR::Reg current_reg{target_reg};

    while (pos >= block_begin) {
        const u64 next_insn{env.ReadInstruction(pos.Offset())};
        const Opcode next_opcode{Decode(next_insn)};
        const Encoding enc{next_insn};

        // Skip arithmetic that reads from our current register and writes to a new one
        if ((next_opcode == Opcode::IADD_cbuf || next_opcode == Opcode::IADD_imm ||
             next_opcode == Opcode::IADD_reg || next_opcode == Opcode::MOV_reg) &&
            enc.src_reg == current_reg) {
            current_reg = enc.dest_reg;
            --pos;
            continue;
        }
        break;
    }

    return current_reg;
}

} // Anonymous namespace

std::optional<IndirectBranchTableInfo> TrackIndirectBranchTable(Environment& env, Location brx_pos,
                                                                Location block_begin) {
    const u64 brx_insn{env.ReadInstruction(brx_pos.Offset())};
    const Opcode brx_opcode{Decode(brx_insn)};
    if (brx_opcode != Opcode::BRX && brx_opcode != Opcode::JMX) {
        throw LogicError("Tracked instruction is not BRX or JMX");
    }
    const IR::Reg brx_reg{Encoding{brx_insn}.src_reg};
    const s32 brx_offset{static_cast<s32>(Encoding{brx_insn}.brx_offset)};

    Location pos{brx_pos};
    --pos; // Move back from BRX/JMX instruction

    // Try standard LDC first
    const std::optional<u64> ldc_insn{TrackLDC(env, block_begin, pos, brx_reg)};
    if (!ldc_insn) {
        LOG_WARNING(Shader,
            "Failed to track indirect branch: LDC not found for BRX/JMX at offset 0x{:x}, "
            "expected register {}",
            brx_pos.Offset(), static_cast<int>(brx_reg));
        return std::nullopt;
    }

    const LDC::Encoding ldc{*ldc_insn};
    const u32 cbuf_index{static_cast<u32>(ldc.index)};
    const u32 cbuf_offset{static_cast<u32>(static_cast<s32>(ldc.offset.Value()))};
    IR::Reg current_reg{ldc.src_reg};

    // Skip MOV instructions after LDC
    current_reg = SkipMovInstructions(env, block_begin, pos, current_reg);

    // SHL is typically present but may be skipped in some patterns
    // Try to find it; if not found, we might still succeed
    const std::optional<u64> shl_insn{TrackSHL(env, block_begin, pos, current_reg)};
    IR::Reg pre_bounds_reg{current_reg};

    if (shl_insn) {
        const Encoding shl{*shl_insn};
        pre_bounds_reg = shl.src_reg;
        // Skip MOV after SHL
        pre_bounds_reg = SkipMovInstructions(env, block_begin, pos, pre_bounds_reg);
    } else {
        LOG_DEBUG(Shader, "No SHL found after LDC at offset 0x{:x}, continuing without it",
                  brx_pos.Offset());
    }

    // Skip additional arithmetic (IADD, etc.) before bounds check
    pre_bounds_reg = SkipArithmetic(env, block_begin, pos, pre_bounds_reg);

    // Look for bounds-checking instruction (IMNMX)
    const std::optional<u64> imnmx_insn{TrackIMNMX(env, block_begin, pos, pre_bounds_reg)};
    if (!imnmx_insn) {
        LOG_WARNING(Shader,
            "Failed to track indirect branch: IMNMX not found for BRX/JMX at offset 0x{:x}, "
            "last tracked register {}",
            brx_pos.Offset(), static_cast<int>(pre_bounds_reg));
        return std::nullopt;
    }

    const Encoding imnmx{*imnmx_insn};
    if (imnmx.is_negative != 0) {
        LOG_WARNING(Shader,
            "Failed to track indirect branch: IMNMX at offset 0x{:x} has negative flag set",
            brx_pos.Offset());
        return std::nullopt;
    }

    const u32 imnmx_immediate{static_cast<u32>(imnmx.immediate.Value())};
    LOG_INFO(Shader,
        "Successfully tracked indirect branch table: cbuf[{}] at offset 0x{:x}, {} entries, "
        "branch_offset=0x{:x}",
        cbuf_index, cbuf_offset, imnmx_immediate + 1, static_cast<u32>(brx_offset));

    return IndirectBranchTableInfo{
        .cbuf_index = cbuf_index,
        .cbuf_offset = cbuf_offset,
        .num_entries = imnmx_immediate + 1,
        .branch_offset = brx_offset,
        .branch_reg = brx_reg,
    };
}

} // namespace Shader::Maxwell

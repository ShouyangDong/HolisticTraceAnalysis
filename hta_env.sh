#!/bin/bash
# HolisticTraceAnalysis 环境配置 (通过 PYTHONPATH 替代 pip install)
# 使用方法:
#   source env.sh                    # 在当前 shell 加载
#   或放到 ~/.bashrc: source /path/to/HolisticTraceAnalysis/env.sh

HTA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Python path ──────────────────────────────────────────────────────────
export PYTHONPATH="$HTA_ROOT:$PYTHONPATH"
echo "✅ HTA: $HTA_ROOT (via PYTHONPATH)"

# ── 可选: 如果 hta 依赖 pandas, numpy 等 ────────────────────────────────
# pip install pandas numpy  # 如果还没装
#
# ── 验证 ─────────────────────────────────────────────────────────────────
if python3 -c "from hta.trace_analysis import TraceAnalysis; print('HTA OK')" 2>/dev/null; then
    echo "✅ HTA import 验证通过"
else
    echo "⚠️  HTA import 失败，检查依赖: pip install pandas numpy"
fi

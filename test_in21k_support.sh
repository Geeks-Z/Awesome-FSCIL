#!/bin/bash
# 快速测试脚本 - 验证 IN21K backbone 是否正确加载

echo "=========================================="
echo "方案A IN21K支持 - 快速验证测试"
echo "=========================================="

# 设置测试参数（只运行1个epoch进行快速验证）
TEST_EPOCH=1

# 测试 1: CODA-Prompt + IN21K
echo ""
echo "测试 1/4: CODA-Prompt with IN21K"
echo "------------------------------------------"
cd /home/team/zhaohongwei/Code/Research/Awesome-FSCIL/CIL_Model
# 临时修改配置文件的epoch为1（仅用于测试）
python main.py --config=./configs/coda_prompt/coda_prompt_cub_B100_Inc10_in21k.json 2>&1 | head -100

# 测试 2: L2P + IN21K
echo ""
echo "测试 2/4: L2P with IN21K"
echo "------------------------------------------"
python main.py --config=./configs/l2p/l2p_cub_B100_Inc10_in21k.json 2>&1 | head -100

# 测试 3: ASP + IN21K
echo ""
echo "测试 3/4: ASP with IN21K"
echo "------------------------------------------"
cd ../FSCIL-ASP-main
python main.py --config=./configs/asp_cub_B100_Inc10_in21k.json 2>&1 | head -100

# 测试 4: SEC + IN21K
echo ""
echo "测试 4/4: SEC with IN21K"
echo "------------------------------------------"
cd ../SEC-Prompt-main
python main.py --config=./exps/SEC-Prompt_cub_B100_Inc10_in21k.json 2>&1 | head -100

echo ""
echo "=========================================="
echo "测试完成！请检查上述输出中是否有："
echo "1. 模型成功加载 (out_dim=768)"
echo "2. 没有 ValueError 或 RuntimeError"
echo "3. 训练能够正常开始"
echo "=========================================="

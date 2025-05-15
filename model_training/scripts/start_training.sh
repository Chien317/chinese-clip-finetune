#!/bin/bash

# 训练启动和监控脚本
# 用于启动训练流程并提供自动重启和监控功能

# 切换到项目根目录
cd $(dirname $0)/..
WORKDIR=$(pwd)
echo "当前工作目录: $WORKDIR"

# 创建日志目录
mkdir -p logs

# 定义错误处理函数
handle_error() {
    echo "[$(date)] 错误: $1" | tee -a logs/error.log
    echo "[$(date)] 尝试重新启动训练..." | tee -a logs/error.log
}

# 定义训练完成的邮件通知函数
send_completion_email() {
    if [ -n "$EMAIL" ]; then
        echo "[$(date)] 训练完成，正在发送通知邮件到 $EMAIL..." | tee -a logs/training.log
        echo "Chinese-CLIP 模型训练已完成。请登录服务器检查结果。" | mail -s "训练完成通知" $EMAIL
    fi
}

# 定义监控函数
monitor_training() {
    local pid=$1
    while kill -0 $pid 2>/dev/null; do
        # 获取最新GPU使用情况
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader | \
            tee -a logs/gpu_usage.log
        
        # 检查是否有在experiments目录下产生新文件
        ls -la experiments/ > logs/experiments_latest.log
        
        # 每5分钟检查一次
        sleep 300
    done
    
    echo "[$(date)] 训练进程 $pid 已结束。" | tee -a logs/training.log
}

# 定义训练流程函数
run_training() {
    echo "[$(date)] 开始执行训练流程..." | tee -a logs/training.log
    
    # 运行训练流水线
    bash scripts/train_pipeline.sh > logs/train_pipeline.log 2>&1 &
    local train_pid=$!
    
    echo "[$(date)] 训练进程已启动，PID: $train_pid" | tee -a logs/training.log
    
    # 在后台监控训练进程
    monitor_training $train_pid &
    local monitor_pid=$!
    
    # 等待训练完成
    wait $train_pid
    local train_status=$?
    
    # 停止监控
    kill $monitor_pid 2>/dev/null
    
    # 检查训练是否成功完成
    if [ $train_status -eq 0 ]; then
        echo "[$(date)] 训练成功完成!" | tee -a logs/training.log
        send_completion_email
        return 0
    else
        handle_error "训练过程异常终止，状态码: $train_status"
        return 1
    fi
}

# 主函数
main() {
    echo "[$(date)] ==== Chinese-CLIP 训练流程启动 ====" | tee -a logs/training.log
    
    # 设置接收邮件通知的邮箱地址（可选）
    export EMAIL=$1
    
    # 最大重试次数
    local max_retries=3
    local retry_count=0
    local success=false
    
    # 尝试运行训练，出错时自动重试
    while [ $retry_count -lt $max_retries ] && [ "$success" != "true" ]; do
        echo "[$(date)] 尝试训练 #$((retry_count+1))..." | tee -a logs/training.log
        
        if run_training; then
            success=true
        else
            retry_count=$((retry_count+1))
            if [ $retry_count -lt $max_retries ]; then
                echo "[$(date)] 等待60秒后重试..." | tee -a logs/training.log
                sleep 60
            fi
        fi
    done
    
    if [ "$success" = "true" ]; then
        echo "[$(date)] ==== 训练成功完成! ====" | tee -a logs/training.log
    else
        echo "[$(date)] ==== 训练失败，已达到最大重试次数 ====" | tee -a logs/training.log
    fi
}

# 检查参数
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "用法: $0 [email@example.com]"
    echo "  参数:"
    echo "    email@example.com - 可选，训练完成后发送通知邮件的地址"
    exit 0
fi

# 执行主函数
main "$@" 
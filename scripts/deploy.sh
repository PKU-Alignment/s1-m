# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=4,5,6,7

vllm serve PKU-Alignment/s1-m_7b_beta \
--served-model-name s1-m_7b_beta \
--port 8000 \
--host 0.0.0.0 \
--tensor-parallel-size 4 \
--gpu-memory-utilization 0.8 \
--limit-mm-per-prompt image=10,video=10 \
--chat-template qwen2vl_test-scaling.jinja \
--enable-prefix-caching \
--dtype bfloat16 \

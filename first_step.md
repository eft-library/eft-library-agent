# 설계

LLM Server

- Ubuntu 22.04
- Ollama
- Qwen3 8B
- Embedding: bge-m3

UI Server (기존 서버)

- Rocky Linux 10
- MCP Server

## Ollama / Qwen3 8B 설치

```shell
# Ollama 설치
curl -fsSL https://ollama.com/install.sh | sh

# 설치 끝나고 해당 문구 뜨면 재부팅 >>> Reboot to complete NVIDIA CUDA driver install.
sudo reboot

# CUDA 드라이버 확인 
nvidia-smi 

# Ollama 서비스 확인 
systemctl status ollama

# Override 폴더 생성하기 - 외부 접속용
sudo mkdir -p /etc/systemd/system/ollama.service.d

# 내용 적기
sudo vi /etc/systemd/system/ollama.service.d/override.conf
[Service]
Environment="OLLAMA_HOST=0.0.0.0:999999"

# 환경 수정
echo 'export OLLAMA_HOST=http://localhost:999999' >> ~/.bashrc
source ~/.bashrc

# 재실행 후 확인
sudo systemctl daemon-reload
sudo systemctl restart ollama
sudo systemctl status ollama

# Qwen3 8B 내려받기
ollama pull qwen3:8b

# 임베딩 모델 bge-m3 내려받기
ollama pull bge-m3

# console로 테스트 해보기 - 인사 해보고 잘 되면 종료
ollama run qwen3:8b 

# 외부 접근 확인
curl http://server-ip:8000/api/chat -d '{
  "model": "qwen3:8b",
  "messages": [{"role": "user", "content": "안녕"}],
  "stream": false
}'
```

## 다행인 점

Ollama 기본값이 **Q4_K_M** 양자화

```shell
# 확인하기
curl http://localhost:8000/api/show -d '{"name": "qwen3:8b"}' | grep -i quant

# 값 확인
{"quantization_level":"Q4_K_M"}
```
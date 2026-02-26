# Ubuntu 22.04 설치 후 1차 환경 세팅

```shell
# / 용량 늘리기
sudo lvextend -r -l +100%FREE /dev/ubuntu-vg/ubuntu-lv

# 패키지 설치
apt update
apt install openssh-server -y
apt install net-tools

# 외부 ip 확인
curl ifcomfig.me

# ssh port 변경
vi /etc/ssh/sshd_config
Port 99999

# Port Open
ufw allow 99999 (SSH 사용할 포트 입력)
ufw allow 999999 (Ollama 사용할 포트 입력)

# ssh restart
systemctl restart ssh
```
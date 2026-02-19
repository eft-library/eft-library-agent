# Ubuntu 22.04 설치 후 1차 환경 세팅

## / 용량 늘리기

sudo lvextend -r -l +100%FREE /dev/ubuntu-vg/ubuntu-lv

## 패키지 설치

apt update
apt install openssh-server -y
apt install net-tools

## 외부 ip 확인

curl ifcomfig.me

## ssh port 변경

vi /etc/ssh/sshd_config
Port 1441

## Port 개발

ufw allow 1441
ufw allow 8000

## ssh restart

systemctl restart ssh

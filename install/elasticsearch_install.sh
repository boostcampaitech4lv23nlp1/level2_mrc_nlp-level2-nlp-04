apt-get update && apt-get -y install sudo
sudo apt update
sudo apt install apt-transport-https
sudo apt install openjdk-8-jdk

apt-get install -y gnupg2
wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | sudo apt-key add -
echo "deb https://artifacts.elastic.co/packages/7.x/apt stable main" | sudo tee -a /etc/apt/sources.list.d/elastic-7.x.list

sudo apt update
sudo apt install elasticsearch
service elasticsearch start

# nori 형태소분석기 설치
cd /usr/share/elasticsearch
bin/elasticsearch-plugin install analysis-nori
service elasticsearch restart

# Python Library 설치
pip install elasticsearch
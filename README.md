# Deep learning site

**Requirements**

Install Docker: https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-18-04

Install Git: https://www.digitalocean.com/community/tutorials/how-to-install-git-on-ubuntu-18-04-quickstart

## Usage

**Step 1. Clone the repository**
```
git clone https://github.com/zinzinhust96/deep-learning-site.git
```

**Step 2. Install the dependencies**
```
cd deep-learning-site
pip3 install -r requirements.txt
```

**Step 3. Copy model files and psiblast software to project directory (./deep-learning-site) from https://drive.google.com/drive/folders/1ub0zkwnQbDLAug_GeHVajsSCGh439FQn?usp=sharing**

**Step 4. To run the project locally, enter the command**
```
python3 main.py
```

**Step 5. To run the production server, we have to build a docker container from image first**
```
docker build . -t deep-learning-site:latest
```

**Step 6: To run the container in the background**
```
docker run --env-file ./env.list -it -d --name dl-site -p 80:5000 -v $PWD/ncbi-blast-2.9.0+:/opt/app/ncbi-blast-2.9.0+ -v $PWD/problem-1:/opt/app/problem-1 -v $PWD/problem-2:/opt/app/problem-2 deep-learning-site:latest
```
(-v specifies the volume mounting to the folder in the project directory to be included in the container, which is too large if we actually include them. If you have more model files, you have to specify exactly like so)

**If you have make changes and want to redeploy the container**
```
git pull
docker ps -a      # and then grab the container_id
docker stop $(container_id)
docker rm $(container_id)
# redo step 5 and 6
```

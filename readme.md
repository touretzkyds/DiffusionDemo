# Jetstream2 Kubernetes Deployment Guide

## 1. Setup OpenRC with Unrestricted Access
- Follow the tutorial: [Jetstream2 OpenRC Documentation](https://docs.jetstream-cloud.org/ui/cli/auth/?h=unrestricted#using-the-horizon-dashboard-to-generate-openrcsh).
- Example openrc.sh:
```bash
export OS_AUTH_TYPE=v3applicationcredential
export OS_AUTH_URL=https://js2.jetstream-cloud.org:5000/v3/
export OS_IDENTITY_API_VERSION=3
export OS_REGION_NAME="IU"
export OS_INTERFACE=public
export OS_APPLICATION_CREDENTIAL_ID=YOUR_CREDENTIAL_ID  # See tutorial
export OS_APPLICATION_CREDENTIAL_SECRET=YOUR_SECRET     # Escape $ symbols as needed
export OS_APPLICATION_CREDENTIAL_NAME=$OS_APPLICATION_CREDENTIAL_ID
export PROJ="xxx000000"                                 # Your allocation id
export CLUSTER=diffusion-demo-deployment                # Deployment name
export K8S_CLUSTER_NAME=diffusion-demo                   # Cluster name
```

## 2. Environment Setup
- Create a new conda environment and install packages:
```bash
conda create -n jetstream2
conda activate jetstream2
pip install python-openstackclient python-magnumclient
```
Also, install:
- kubectl (see https://kubernetes.io/docs/tasks/tools/; this tutorial used v1.26)
- helm (see https://helm.sh/docs/intro/install/; this tutorial used v3.8.1)

Remember to source your openrc.sh:
```bash
source openrc.sh
```

## 3. Clone Repository and Navigation
Clone the repository and navigate to the working directory:
```bash
git clone https://github.com/zonca/jupyterhub-deploy-kubernetes-jetstream
cd jupyterhub-deploy-kubernetes-jetstream/kubernetes_magnum
```
Also refer to the tutorial:  
https://www.zonca.dev/posts/2024-12-11-jetstream_kubernetes_magnum

## 4. GPU Cluster Configuration (Optional)
- If you want GPU support, override the default values:
```bash
# Override default template values for GPU clusters
FLAVOR="g3.xl"
TEMPLATE="kubernetes-1-30-jammy"
MASTER_FLAVOR="m3.medium"
DOCKER_VOLUME_SIZE_GB=10
```

## 5. Cluster Creation
- Standard cluster creation:
```bash
openstack coe cluster create --cluster-template $TEMPLATE \
    --master-count $N_MASTER --node-count $N_NODES \
    --master-flavor $MASTER_FLAVOR --flavor $FLAVOR \
    --docker-volume-size $DOCKER_VOLUME_SIZE_GB \
    --labels auto_scaling_enabled=true \
    --labels min_node_count=1 \
    --labels max_node_count=5 \
    $K8S_CLUSTER_NAME
```
- For a larger boot volume (100GB):
```bash
openstack coe cluster create --cluster-template $TEMPLATE \
    --master-count $N_MASTER --node-count $N_NODES \
    --master-flavor $MASTER_FLAVOR --flavor $FLAVOR \
    --docker-volume-size $DOCKER_VOLUME_SIZE_GB \
    --labels auto_scaling_enabled=true \
    --labels min_node_count=1 \
    --labels max_node_count=5 \
    --labels boot_volume_size=100 \
    $K8S_CLUSTER_NAME
```

## 6. Configure Cluster Access
```bash
openstack coe cluster config $K8S_CLUSTER_NAME --force
export KUBECONFIG=$(pwd)/config
chmod 600 config
kubectl get nodes  # Verify nodes are running
```

## 7. MIG GPU Configuration
To efficiently use GPUs, enable MIG. Note there is a known bug: only the g3.xl flavor is supported (track issue: https://gitlab.com/jetstream-cloud/jetstream2/project-management/-/issues/229).  
Follow NVIDIA’s MIG tutorial here:  
https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-operator-mig.html#configuring-mig-profiles

For enabling MIG on a worker node:
```bash
kubectl label nodes <node-name> nvidia.com/mig.config=all-2g.10gb --overwrite
kubectl get node <node-name> -o=jsonpath='{.metadata.labels}' | jq .
```
Supported MIG configurations for NVIDIA A100-SXM4-40GB (for Jetstream2):
- MIG 1g.5gb
- MIG 1g.10gb
- MIG 2g.10gb
- MIG 3g.20gb
- MIG 4g.20gb
- MIG 7g.40gb

You can find more details here:  
https://docs.nvidia.com/datacenter/tesla/mig-user-guide/

For a single MIG strategy (applying the same config to all sub GPUs), you should eventually see:
```
"nvidia.com/mig.config": "all-1g.10gb",
"nvidia.com/mig.config.state": "success"
```
instead of "pending".

## 8. Application Deployment

### Deployment File
Create a deployment file (e.g., deployment.yaml):
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: diffusion-demo-deployment
  labels:
    app: diffusion-demo
spec:
  replicas: 1
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1       
      maxUnavailable: 0   
  selector:
    matchLabels:
      app: diffusion-demo
  template:
    metadata:
      labels:
        app: diffusion-demo
    spec:
      containers:
      - name: diffusion-demo
        image: akameswa/diffusion-demo-cuda-slim:latest
        ports:
        - containerPort: 7860
        resources:
          requests:
            cpu: "4"
            memory: "15Gi"
            nvidia.com/gpu: 1
          limits:
            cpu: "4"
            memory: "15Gi"
            nvidia.com/gpu: 1
```

### Service File
Create a service file (e.g., service.yaml) to expose the pod internally:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: diffusion-demo-service
spec:
  type: ClusterIP
  selector:
    app: diffusion-demo
  ports:
    - protocol: TCP
      port: 80
      targetPort: 7860
```

## 9. Dockerization
The application is a Gradio app. Dockerization follows this tutorial:  
https://www.gradio.app/guides/deploying-gradio-with-docker

Example Dockerfile:
```dockerfile
# https://huggingface.co/spaces/SpacesExamples/Gradio-Docker-Template-nvidia-cuda/blob/main/Dockerfile
# https://www.gradio.app/guides/deploying-gradio-with-docker
# To run locally:
# docker run -p 7860:7860 --rm --runtime=nvidia --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface akameswa/diffusion-demo-cuda-slim:latest
FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime

ENV GRADIO_SERVER_NAME=0.0.0.0 
EXPOSE 7860

COPY DiffusionDemo/ DiffusionDemo/

RUN apt-get update && \
    apt-get install -y curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir -r DiffusionDemo/requirements.txt -q

CMD ["python3", "DiffusionDemo/run.py"]
```
Build the Docker image:
```bash
docker build -t gradio-app .
```
Push the image to Docker Hub (login first if required). Follow the tutorial:  
https://docs.docker.com/reference/cli/docker/image/push/
```bash
docker push your_dockerhub_username/gradio-app
```
You can test it locally:
```bash
docker run -p 7860:7860 --rm --runtime=nvidia --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface your_dockerhub_username/gradio-app
```
Note: The deployment runs on partial GPU.

## 10. Exposing the Application via Ingress

### Ingress Controller
Install the NGINX Ingress Controller:
```bash
helm upgrade --install ingress-nginx ingress-nginx \
    --repo https://kubernetes.github.io/ingress-nginx \
    --namespace ingress-nginx --create-namespace
```
Ensure it's running:
```bash
kubectl get svc -n ingress-nginx ingress-nginx-controller
```
Get the controller's IP and create a DNS record:
```bash
openstack recordset create $PROJ.projects.jetstream-cloud.org. k8s --type A --record $IP --ttl 3600
```

### Ingress Resource
Create an ingress file (e.g., ingress.yaml):
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: diffusion-demo-ingress
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  ingressClassName: nginx
  rules:
  - hoyour: your_clustername.your_id_project.projects.jetstream-cloud.org
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: diffusion-demo-service
            port:
              number: 80
  tls:
  - hosts:
    - your_clustername.your_id_project.projects.jetstream-cloud.org
    secretName: diffusion-demo-tls
```

### HTTPS with Cert-Manager
Apply cert-manager:
```bash
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.16.2/cert-manager.yaml
```
Create a ClusterIssuer (e.g., cluster-issuer.yaml):
```yaml
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt
spec:
  acme:
    # Replace with your email address for notifications about expiring certificates.
    email: your-email@domain.com
    server: https://acme-v02.api.letsencrypt.org/directory
    privateKeySecretRef:
      name: cluster-issuer-account-key
    solvers:
    - http01:
        ingress:
          class: nginx
```
Apply the ClusterIssuer:
```bash
kubectl create -f cluster-issuer.yaml
```
Finally, create and apply a Certificate (e.g., certificate.yaml):
```yaml
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: diffusion-demo-certificate
  namespace: default
spec:
  secretName: diffusion-demo-tls
  issuerRef:
    name: letsencrypt
    kind: ClusterIssuer        
  commonName: your_clustername.your_id_project.projects.jetstream-cloud.org
  dnsNames:
  - your_clustername.your_id_project.projects.jetstream-cloud.org
```
Then run:
```bash
kubectl create -f certificate.yaml
```

## Notes
- Replace all placeholders (e.g., YOUR_CREDENTIAL_ID, YOUR_SECRET, your_dockerhub_username, your-email@domain.com, and your domain names) with your actual values.
- Follow the linked tutorials for further details.

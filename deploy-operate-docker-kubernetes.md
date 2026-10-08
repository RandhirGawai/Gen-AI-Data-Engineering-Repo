# Deploying and Operating Services in Production (Docker + Kubernetes)

## Question

> **QuickCart** is an e-commerce company with a Python **FastAPI** `order-service`. It currently runs on a single VM with `python app.py`, which causes downtime during releases, no automatic recovery when it crashes, and no visibility when it is slow.
>
> You are asked to **containerize the service with Docker, deploy it to Kubernetes, and operate it in production** so that it supports:
> - zero-downtime releases and fast rollback
> - automatic scaling during sale events
> - secure handling of secrets and configuration
> - monitoring, logging, and alerting
>
> Explain every step: from the Dockerfile to the Kubernetes manifests, CI/CD, observability, and day-2 operations. Also describe how you would troubleshoot a failing deployment.

---

## Answer Overview

```text
Developer -> Git push -> CI (test, build, scan) -> Container Registry
                                                        |
                                                        v
                         Kubernetes cluster (Deployment + Service + Ingress + HPA)
                                                        |
                                    Metrics / Logs / Alerts (Prometheus, Grafana, Loki)
```

| Layer | Purpose | Key objects / tools |
|---|---|---|
| **Container** | Reproducible, portable package | Dockerfile, image registry |
| **Orchestration** | Run, heal, scale, update | Deployment, Service, HPA |
| **Config and secrets** | Separate config from code | ConfigMap, Secret |
| **Networking** | Expose traffic safely | Service, Ingress, TLS |
| **Delivery** | Automate releases | CI/CD, rolling update, rollback |
| **Operations** | Observe and respond | Probes, metrics, logs, alerts |

---

## Part A: Docker

### Step 1: Prepare the application for containers

Follow the twelve-factor approach:

- Read configuration from **environment variables**, not files in the image
- Log to **stdout/stderr**
- Expose a **health endpoint**
- Handle `SIGTERM` for graceful shutdown (uvicorn does this by default)

```python
# app/main.py
import os
from fastapi import FastAPI

app = FastAPI()

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/readyz")
def readyz():
    # Check dependencies here (DB, cache). Return 503 if not ready.
    return {"status": "ready"}

@app.get("/orders/{order_id}")
def get_order(order_id: str):
    return {"order_id": order_id, "env": os.getenv("APP_ENV", "dev")}
```

### Step 2: Write a production-grade Dockerfile

Use a **multi-stage build**, a slim base image, pinned versions, and a **non-root user**.

```dockerfile
# ---------- Build stage ----------
FROM python:3.12-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------- Runtime stage ----------
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Non-root user
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app

COPY --from=builder /install /usr/local
COPY app ./app

USER 10001
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Add a `.dockerignore`:

```text
.git
__pycache__/
*.pyc
.venv/
tests/
.env
```

### Step 3: Build, run, and test locally

```bash
docker build -t quickcart/order-service:1.0.0 .
docker run --rm -p 8000:8000 -e APP_ENV=local quickcart/order-service:1.0.0
curl http://localhost:8000/healthz
```

### Step 4: Scan and push to a registry

```bash
# Vulnerability scan (example with Trivy)
trivy image quickcart/order-service:1.0.0

# Tag and push (ACR / ECR / GHCR / Docker Hub)
docker tag quickcart/order-service:1.0.0 registry.quickcart.example.com/order-service:1.0.0
docker push registry.quickcart.example.com/order-service:1.0.0
```

**Image best practices**

- Never use `latest` in production; use immutable tags (version or git SHA)
- Keep images small; no build tools in the runtime stage
- Scan in CI and fail the build on critical CVEs
- Never bake secrets into the image

---

## Part B: Kubernetes Deployment

### Step 5: Create a namespace

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: quickcart-prod
```

```bash
kubectl apply -f namespace.yaml
```

### Step 6: Configuration and secrets

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: order-service-config
  namespace: quickcart-prod
data:
  APP_ENV: "production"
  LOG_LEVEL: "info"
```

```bash
# Create the secret from the CLI (do not commit real secrets to Git)
kubectl create secret generic order-service-secret \
  --namespace quickcart-prod \
  --from-literal=DB_PASSWORD='change-me'
```

In production, use an external secret manager (Azure Key Vault, AWS Secrets Manager, HashiCorp Vault) with the External Secrets Operator or the CSI driver, and enable encryption at rest for Secrets.

### Step 7: Deployment (the core object)

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: order-service
  namespace: quickcart-prod
  labels:
    app: order-service
spec:
  replicas: 3
  revisionHistoryLimit: 5
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 0        # never drop below desired capacity
      maxSurge: 1              # add one new pod at a time
  selector:
    matchLabels:
      app: order-service
  template:
    metadata:
      labels:
        app: order-service
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 10001
      terminationGracePeriodSeconds: 30
      containers:
        - name: order-service
          image: registry.quickcart.example.com/order-service:1.0.0
          ports:
            - containerPort: 8000
          envFrom:
            - configMapRef:
                name: order-service-config
          env:
            - name: DB_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: order-service-secret
                  key: DB_PASSWORD
          resources:
            requests:
              cpu: "250m"
              memory: "256Mi"
            limits:
              cpu: "500m"
              memory: "512Mi"
          startupProbe:
            httpGet:
              path: /healthz
              port: 8000
            failureThreshold: 30
            periodSeconds: 2
          readinessProbe:
            httpGet:
              path: /readyz
              port: 8000
            periodSeconds: 5
            failureThreshold: 3
          livenessProbe:
            httpGet:
              path: /healthz
              port: 8000
            periodSeconds: 10
            failureThreshold: 3
          lifecycle:
            preStop:
              exec:
                command: ["sleep", "5"]   # let the load balancer drain
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities:
              drop: ["ALL"]
```

**Why each part matters**

| Setting | Reason |
|---|---|
| `replicas: 3` | High availability |
| `maxUnavailable: 0` | Zero-downtime rolling update |
| **requests / limits** | Scheduling, fair sharing, protection from noisy neighbors |
| **startupProbe** | Gives slow-starting apps time before liveness checks begin |
| **readinessProbe** | Pod receives traffic only when ready |
| **livenessProbe** | Restarts a hung container |
| `preStop` + grace period | Graceful shutdown without dropped requests |
| `securityContext` | Least privilege at runtime |

### Step 8: Service (stable internal endpoint)

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: order-service
  namespace: quickcart-prod
spec:
  type: ClusterIP
  selector:
    app: order-service
  ports:
    - port: 80
      targetPort: 8000
```

### Step 9: Ingress with TLS (external access)

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: order-service
  namespace: quickcart-prod
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  ingressClassName: nginx
  tls:
    - hosts: ["api.quickcart.example.com"]
      secretName: order-service-tls
  rules:
    - host: api.quickcart.example.com
      http:
        paths:
          - path: /orders
            pathType: Prefix
            backend:
              service:
                name: order-service
                port:
                  number: 80
```

### Step 10: Autoscaling and disruption protection

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: order-service
  namespace: quickcart-prod
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: order-service
  minReplicas: 3
  maxReplicas: 15
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
---
# pdb.yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: order-service
  namespace: quickcart-prod
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: order-service
```

The HPA needs the **metrics-server** installed and CPU **requests** set on the pods. The PDB keeps at least two pods up during node drains and upgrades.

### Step 11: Network policy (restrict traffic)

```yaml
# networkpolicy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: order-service
  namespace: quickcart-prod
spec:
  podSelector:
    matchLabels:
      app: order-service
  policyTypes: ["Ingress"]
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: ingress-nginx
      ports:
        - port: 8000
```

### Step 12: Apply and verify

```bash
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f deployment.yaml -f service.yaml -f ingress.yaml
kubectl apply -f hpa.yaml -f pdb.yaml -f networkpolicy.yaml

kubectl -n quickcart-prod rollout status deployment/order-service
kubectl -n quickcart-prod get pods,svc,ingress,hpa
```

---

## Part C: CI/CD

### Step 13: Automate build, test, scan, and deploy

Example GitHub Actions pipeline:

```yaml
# .github/workflows/deploy.yml
name: build-and-deploy
on:
  push:
    branches: [main]

env:
  IMAGE: registry.quickcart.example.com/order-service

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run tests
        run: |
          pip install -r requirements.txt pytest
          pytest

      - name: Build image
        run: docker build -t $IMAGE:${{ github.sha }} .

      - name: Scan image
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ${{ env.IMAGE }}:${{ github.sha }}
          severity: CRITICAL,HIGH
          exit-code: "1"

      - name: Push image
        run: |
          echo "${{ secrets.REGISTRY_PASSWORD }}" | docker login registry.quickcart.example.com -u "${{ secrets.REGISTRY_USER }}" --password-stdin
          docker push $IMAGE:${{ github.sha }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Deploy to Kubernetes
        run: |
          kubectl -n quickcart-prod set image deployment/order-service \
            order-service=$IMAGE:${{ github.sha }}
          kubectl -n quickcart-prod rollout status deployment/order-service --timeout=180s
```

(Cluster credentials are configured via a kubeconfig secret or OIDC federation in a prior step.)

**Better practice: GitOps.** Keep manifests in a Git repo and let **Argo CD** or **Flux** sync the cluster to Git. Packaging with **Helm** or **Kustomize** gives you per-environment values (dev, staging, prod).

### Release strategies

| Strategy | How | When to use |
|---|---|---|
| **Rolling update** | Default; replace pods gradually | Most services |
| **Blue/Green** | Run two full environments, switch traffic | Instant switch and rollback |
| **Canary** | Send 5-10% of traffic to the new version first | Risky changes (Argo Rollouts, Istio, NGINX) |

---

## Part D: Operating in Production (Day-2)

### Step 14: Observability

**Logs:** write structured JSON logs to stdout, collected by Fluent Bit / Promtail into Loki, ELK, or a cloud logging service.

```bash
kubectl -n quickcart-prod logs deployment/order-service --tail=100 -f
kubectl -n quickcart-prod logs <pod> --previous      # logs of the crashed container
```

**Metrics:** expose `/metrics` (for example with `prometheus-fastapi-instrumentator`) and scrape with Prometheus; visualize in Grafana.

Track the **four golden signals**:

| Signal | Example metric |
|---|---|
| Latency | p95 / p99 request duration |
| Traffic | Requests per second |
| Errors | 5xx rate |
| Saturation | CPU, memory, queue depth |

**Traces:** instrument with OpenTelemetry and send to Jaeger / Tempo / Azure Monitor.

**Alerts (examples)**

- 5xx error rate above 2% for 5 minutes
- p95 latency above 800 ms for 10 minutes
- Pod restarts above 3 in 15 minutes
- HPA at max replicas for over 15 minutes
- Deployment has fewer available replicas than desired

### Step 15: Rollback

```bash
kubectl -n quickcart-prod rollout history deployment/order-service
kubectl -n quickcart-prod rollout undo deployment/order-service
kubectl -n quickcart-prod rollout undo deployment/order-service --to-revision=3
```

### Step 16: Scaling and capacity

```bash
kubectl -n quickcart-prod scale deployment/order-service --replicas=6   # manual (HPA may override)
kubectl -n quickcart-prod top pods                                       # needs metrics-server
```

- Set requests from real usage data, then tune
- Add a **Cluster Autoscaler** (or Karpenter) so nodes scale with pods
- Load-test before sale events

### Step 17: Security hardening

- Run as non-root, read-only filesystem, drop all capabilities
- RBAC with least-privilege service accounts
- NetworkPolicies (default deny, then allow)
- Pod Security Standards ("restricted") on the namespace
- Image signing and admission policies (Cosign, Kyverno, OPA Gatekeeper)
- Regular base image and cluster version updates

### Step 18: Reliability and backup

- Spread replicas across zones with `topologySpreadConstraints`
- Back up persistent data and etcd; test restores
- Define SLOs (for example 99.9% availability) and error budgets
- Write runbooks and run incident reviews (blameless postmortems)

---

## Part E: Troubleshooting Cheat Sheet

| Symptom | Likely cause | How to investigate |
|---|---|---|
| `ImagePullBackOff` | Wrong image name/tag, missing registry credentials | `kubectl describe pod <pod>` (Events section) |
| `CrashLoopBackOff` | App crashing on start, bad config or secret | `kubectl logs <pod> --previous` |
| `Pending` pod | Not enough CPU/memory, taints, no matching node | `kubectl describe pod <pod>` |
| `OOMKilled` | Memory limit too low or memory leak | `kubectl describe pod`, raise limit or fix leak |
| Pod `Running` but `0/1 Ready` | Readiness probe failing | Check probe path/port and dependency health |
| 502/503 from Ingress | No ready endpoints, wrong service port | `kubectl get endpoints order-service -n quickcart-prod` |
| Rollout stuck | New pods never become ready | `kubectl rollout status`, then describe and logs |
| HPA shows `<unknown>` | metrics-server missing or no CPU requests | Install metrics-server, set `resources.requests` |

Useful commands:

```bash
kubectl -n quickcart-prod describe pod <pod>
kubectl -n quickcart-prod get events --sort-by=.lastTimestamp
kubectl -n quickcart-prod exec -it <pod> -- sh
kubectl -n quickcart-prod port-forward svc/order-service 8080:80
kubectl -n quickcart-prod get deployment order-service -o yaml
```

---

## Quick Recap Checklist

1. Make the app container-friendly (env config, stdout logs, health endpoints)
2. Write a multi-stage, non-root Dockerfile with pinned versions
3. Build, scan, and push an immutable image tag
4. Create the namespace, ConfigMap, and Secret
5. Deploy with probes, resource requests/limits, and a rolling update strategy
6. Expose with Service and Ingress (TLS)
7. Add HPA, PodDisruptionBudget, and NetworkPolicy
8. Automate with CI/CD (or GitOps) and choose a release strategy
9. Add logs, metrics, traces, and alerts
10. Know how to roll back, scale, and troubleshoot
11. Harden security and plan for reliability and backups

---

## Common Pitfalls

- Using the `latest` tag, which makes deployments non-reproducible
- No resource requests/limits, leading to noisy neighbors and failed autoscaling
- Missing readiness probes, so traffic hits pods that are not ready
- Liveness probes that check dependencies, causing restart storms during outages
- Secrets committed to Git or baked into images
- Running as root with a writable filesystem
- Deploying without a tested rollback path
- No alerts, so you learn about outages from customers

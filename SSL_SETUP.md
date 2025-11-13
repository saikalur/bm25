# SSL/HTTPS Setup Guide

This guide walks you through setting up HTTPS for the BakerMatcher application using Let's Encrypt (free SSL certificates) and nginx.

## Prerequisites

- Ubuntu server with domain name pointing to it (`milestone-tracking.cardinaltalent.ai`)
- Ports 80 and 443 open in your firewall
- Root or sudo access

## Step 1: Install nginx

```bash
sudo apt-get update
sudo apt-get install nginx -y
sudo systemctl start nginx
sudo systemctl enable nginx
```

## Step 2: Install Certbot

```bash
sudo apt-get install certbot python3-certbot-nginx -y
```

## Step 3: Configure nginx (Before SSL)

Create nginx configuration file:

```bash
sudo nano /etc/nginx/sites-available/bm25
```

Add this configuration:

```nginx
server {
    listen 80;
    server_name app.bakermatcher.com;

    # Frontend (Vite dev server)
    # Note: Check which port Vite is using (may be 5173, 5174, 5175, etc.)
    location / {
        proxy_pass http://localhost:8801;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }

    # Backend API
    location /api {
        proxy_pass http://localhost:5000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket support for Vite HMR
    location /ws {
        proxy_pass http://localhost:8801;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

Enable the site:

```bash
sudo ln -s /etc/nginx/sites-available/bm25 /etc/nginx/sites-enabled/
sudo nginx -t  # Test configuration
sudo systemctl reload nginx
```

## Step 4: Get SSL Certificate

```bash
sudo certbot --nginx -d milestone-tracking.cardinaltalent.ai
```

Follow the prompts:
- Enter your email address
- Agree to terms
- Choose whether to redirect HTTP to HTTPS (recommended: Yes)

Certbot will automatically:
- Get the certificate
- Configure nginx for HTTPS
- Set up auto-renewal

## Step 5: Update Frontend API URL

Update the frontend to use HTTPS. In `ui/src/App.jsx`, the API_BASE_URL should use HTTPS:

```javascript
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'https://milestone-tracking.cardinaltalent.ai';
```

Or set it via environment variable when building/running.

## Step 6: Verify

1. Access `https://milestone-tracking.cardinaltalent.ai`
2. Check that microphone access works
3. Verify the SSL certificate is valid (lock icon in browser)

## Auto-Renewal

Certbot sets up auto-renewal automatically. Certificates expire every 90 days and will auto-renew.

To test renewal:
```bash
sudo certbot renew --dry-run
```

## Troubleshooting

### Check nginx status
```bash
sudo systemctl status nginx
```

### Check nginx logs
```bash
sudo tail -f /var/log/nginx/error.log
```

### Restart nginx
```bash
sudo systemctl restart nginx
```

### Check if ports are open
```bash
sudo ufw status
sudo ufw allow 80
sudo ufw allow 443
```

## Notes

- The Vite dev server and Flask backend should still run on their original ports (5173 and 5000)
- nginx proxies requests to them
- Users access via HTTPS on port 443 (or 80 which redirects to 443)
- The backend API_BASE_URL in the frontend should be updated to use HTTPS


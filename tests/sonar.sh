#!/bin/bash
# SonarQube analysis script
# Run: ./tests/sonar.sh
# Prerequisites: Docker, npm (for sonar-scanner)

# fail fast
set -euo pipefail

# cd to project root
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "🔍 Starting SonarQube Analysis"

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "⚠️  Docker not found. SonarQube requires Docker to run locally."
    exit 1
fi

# Smart SonarQube container management
echo "🐳 Checking SonarQube container status..."

# Check if container exists (running or stopped)
if docker ps -a --format "{{.Names}}" | grep -q "^sonarqube$"; then
    # Container exists, check if it's running
    if docker ps --format "{{.Names}}" | grep -q "^sonarqube$"; then
        echo "✅ SonarQube container is already running"
        CONTAINER_NEEDS_WAIT=false
    else
        echo "🔄 Starting existing SonarQube container..."
        docker start sonarqube
        CONTAINER_NEEDS_WAIT=true
    fi
else
    # Container doesn't exist, create it
    echo "🚀 Creating new SonarQube container..."
    docker run -d --name sonarqube -p 9000:9000 sonarqube:community
    CONTAINER_NEEDS_WAIT=true
fi

# Wait for SonarQube to be ready (only if we started/restarted the container)
if [ "$CONTAINER_NEEDS_WAIT" = true ]; then
    echo "⏳ Waiting for SonarQube to start (this may take 30-60 seconds)..."
    for i in {1..60}; do
        if curl -s http://localhost:9000/api/system/status | grep -q "UP"; then
            echo "✅ SonarQube is ready!"
            break
        fi
        echo "   Waiting... ($i/60)"
        sleep 2
    done

    if ! curl -s http://localhost:9000/api/system/status | grep -q "UP"; then
        echo "⚠️  SonarQube failed to start properly. Check Docker logs: docker logs sonarqube"
        exit 1
    fi
else
    # Container was already running, just verify it's responsive
    if curl -s http://localhost:9000/api/system/status | grep -q "UP"; then
        echo "✅ SonarQube is ready!"
    else
        echo "⚠️  SonarQube container is running but not responsive. Check Docker logs: docker logs sonarqube"
        exit 1
    fi
fi

# Check for sonar-scanner
if ! command -v sonar-scanner &> /dev/null; then
    echo "📦 Installing sonar-scanner..."
    if command -v npm &> /dev/null; then
        npm install -g sonar-scanner
    else
        echo "⚠️  npm not available. Please install sonar-scanner manually."
        exit 1
    fi
fi

echo "🔍 Running SonarQube analysis..."

# Create basic sonar-project.properties if it doesn't exist in tests directory
if [[ ! -f "tests/sonar-project.properties" ]]; then
    # Check for required environment variables
    if [[ -n "${SONAR_TOKEN:-}" ]]; then
        # Use token-based authentication (preferred)
        cat > tests/sonar-project.properties << 'EOF'
sonar.projectKey=fl-execution-framework
sonar.projectName=FL Execution Framework
sonar.projectVersion=1.0
sonar.host.url=http://localhost:9000
sonar.token=${SONAR_TOKEN}
sonar.sources=.
sonar.python.version=3.10
sonar.python.coverage.reportPaths=logs/coverage.xml
sonar.exclusions=**/logs/**,**/temp/**,**/__pycache__/**
sonar.working.directory=logs/.sonarwork
EOF
        echo "📄 Created tests/sonar-project.properties with token authentication"
    elif [[ -n "${SONAR_USER:-}" && -n "${SONAR_PASS:-}" ]]; then
        # Use username/password authentication
        cat > tests/sonar-project.properties << 'EOF'
sonar.projectKey=fl-execution-framework
sonar.projectName=FL Execution Framework
sonar.projectVersion=1.0
sonar.host.url=http://localhost:9000
sonar.login=${SONAR_USER}
sonar.password=${SONAR_PASS}
sonar.sources=.
sonar.python.version=3.10
sonar.python.coverage.reportPaths=logs/coverage.xml
sonar.exclusions=**/logs/**,**/temp/**,**/__pycache__/**
sonar.working.directory=logs/.sonarwork
EOF
        echo "📄 Created tests/sonar-project.properties with username/password authentication"
    else
        # Fallback configuration without authentication (requires SonarQube configured for anonymous access)
        cat > tests/sonar-project.properties << EOF
sonar.projectKey=fl-execution-framework
sonar.projectName=FL Execution Framework
sonar.projectVersion=1.0
sonar.host.url=http://localhost:9000
sonar.sources=.
sonar.python.version=3.10
sonar.python.coverage.reportPaths=logs/coverage.xml
sonar.exclusions=**/logs/**,**/temp/**,**/__pycache__/**
sonar.working.directory=logs/.sonarwork
EOF
        echo "⚠️  No SONAR_TOKEN or SONAR_USER/SONAR_PASS found."
        echo "   For first-time setup:"
        echo "   1. Visit http://localhost:9000"
        echo "   2. Login with admin/admin (change password when prompted)"
        echo "   3. Generate a token: My Account → Security → Generate Token"
        echo "   4. Set SONAR_TOKEN environment variable"
        echo "   Analysis will likely fail without authentication credentials."
    fi
fi

cd tests
echo "🔍 Starting SonarQube analysis..."
SONAR_OUTPUT=$(mktemp)
if sonar-scanner 2>&1 | tee "$SONAR_OUTPUT"; then
    # Extract dashboard URL from sonar-scanner output
    DASHBOARD_URL=$(grep -o "http://localhost:9000/dashboard?id=[^[:space:]]*" "$SONAR_OUTPUT" || echo "")
    if [[ -n "$DASHBOARD_URL" ]]; then
        echo "✅ SonarQube analysis completed successfully: $DASHBOARD_URL"
    else
        echo "✅ SonarQube analysis completed successfully"
    fi
else
    echo "❌ SonarQube analysis failed (likely authentication issue)"
    echo "   If you see '401 Unauthorized', follow the setup instructions above"
    echo "   Analysis failure won't stop the build process"
fi
rm -f "$SONAR_OUTPUT"
cd ..

echo "✅ SonarQube analysis completed!"
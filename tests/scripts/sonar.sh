#!/bin/sh
# SonarQube analysis script

. "$(dirname "$0")/common.sh"
navigate_to_root

log_info "🔍 Starting SonarQube Analysis"

# Docker & SonarQube server
if ! command_exists docker; then
    log_error "Docker not found. SonarQube requires Docker to run locally."
    exit 1
fi

log_info "🐳 Checking SonarQube container status..."
CONTAINER_NEEDS_WAIT=false
if docker ps -a --format "{{.Names}}" | grep -q "^sonarqube$"; then
    if ! docker ps --format "{{.Names}}" | grep -q "^sonarqube$"; then
        log_info "🔄 Starting existing SonarQube container..."
        docker start sonarqube
        CONTAINER_NEEDS_WAIT=true
    else
        log_info "✅ SonarQube container is already running."
    fi
else
    log_info "🚀 Creating new SonarQube container..."
    docker run -d --name sonarqube -p 9000:9000 sonarqube:community
    CONTAINER_NEEDS_WAIT=true
fi

if [ "$CONTAINER_NEEDS_WAIT" = true ]; then
    log_info "⏳ Waiting for SonarQube to start (this may take up to 60 seconds)..."
    _counter=0
    while [ $_counter -lt 30 ]; do
        if curl -s http://localhost:9000/api/system/status | grep -q "UP"; then
            log_info "✅ SonarQube is ready!"
            break
        fi
        sleep 2
        _counter=$((_counter + 1))
    done
    if ! curl -s http://localhost:9000/api/system/status | grep -q "UP"; then
        log_error "SonarQube failed to start. Check Docker logs: docker logs sonarqube"
        exit 1
    fi
fi

# Sonar scanner
if ! command_exists npm; then
    log_error "npm not found. Please install npm first to install sonar-scanner."
    exit 1
fi

check_and_install_tool "sonar-scanner" "npm install -g sonar-scanner" || {
    log_error "Failed to install sonar-scanner. Please install manually."
    exit 1
}


log_info "🚀 Running SonarQube scanner..."

if sonar-scanner \
    -Dsonar.projectKey=fl-execution-framework \
    -Dsonar.projectName="FL Execution Framework" \
    -Dsonar.projectVersion=1.0 \
    -Dsonar.host.url=http://localhost:9000 \
    -Dsonar.sources=src \
    -Dsonar.tests=tests \
    -Dsonar.python.version=3.10 \
    -Dsonar.python.coverage.reportPaths=tests/logs/coverage.xml \
    -Dsonar.exclusions="**/__pycache__/**,**/.venv/**,**/venv/**,tests/logs/**" \
    -Dsonar.working.directory=tests/logs/.sonarwork; then
    log_info "✅ SonarQube analysis completed successfully."
    log_info "   View the report at: http://localhost:9000/dashboard?id=fl-execution-framework"
else
    log_error "SonarQube analysis failed. Check the output above for details."
    exit 1
fi

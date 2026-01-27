#!/usr/bin/env bash
# Local E2E Test Script for Faz 2
# Usage: ./scripts/local-e2e-test.sh

set -e

echo "=== CopperMind Faz 2 Local E2E Test ==="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check prerequisites
check_prereqs() {
    echo -e "\n${YELLOW}[1/7] Checking prerequisites...${NC}"
    
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Docker not found${NC}"
        exit 1
    fi
    
    if ! command -v psql &> /dev/null; then
        echo -e "${YELLOW}Warning: psql not found - DB verification will be skipped${NC}"
    fi
    
    echo -e "${GREEN}Prerequisites OK${NC}"
}

# Start services
start_services() {
    echo -e "\n${YELLOW}[2/7] Starting PostgreSQL and Redis...${NC}"
    docker-compose up -d postgres redis
    
    # Wait for postgres
    echo "Waiting for PostgreSQL..."
    sleep 5
    
    # Verify
    docker-compose exec -T postgres pg_isready -U coppermind || {
        echo -e "${RED}PostgreSQL not ready${NC}"
        exit 1
    }
    
    docker-compose exec -T redis redis-cli ping || {
        echo -e "${RED}Redis not ready${NC}"
        exit 1
    }
    
    echo -e "${GREEN}Services started${NC}"
}

# Run migration
run_migration() {
    echo -e "\n${YELLOW}[3/7] Running Faz 2 migration...${NC}"
    
    # Use docker exec to run migration
    docker-compose exec -T postgres psql -U coppermind -d coppermind < backend/migrations/001_add_news_raw_processed.sql
    
    echo -e "${GREEN}Migration complete${NC}"
}

# Verify migration
verify_migration() {
    echo -e "\n${YELLOW}[4/7] Verifying migration...${NC}"
    
    # Check tables exist
    docker-compose exec -T postgres psql -U coppermind -d coppermind -c "
        SELECT 
            (SELECT to_regclass('public.news_raw')) as news_raw,
            (SELECT to_regclass('public.news_processed')) as news_processed;
    "
    
    # Check partial unique index
    docker-compose exec -T postgres psql -U coppermind -d coppermind -c "
        SELECT indexname FROM pg_indexes 
        WHERE tablename='news_raw' AND indexname LIKE '%url_hash%';
    "
    
    echo -e "${GREEN}Migration verified${NC}"
}

# Start API and Worker
start_app() {
    echo -e "\n${YELLOW}[5/7] Starting API and Worker...${NC}"
    
    # Use local Python environment
    cd backend
    
    # API in background
    echo "Starting API..."
    DATABASE_URL="postgresql://coppermind:localdev@localhost:5432/coppermind" \
    REDIS_URL="redis://localhost:6379/0" \
    PIPELINE_TRIGGER_SECRET="local-test-secret" \
    uvicorn app.main:app --host 0.0.0.0 --port 8000 &
    API_PID=$!
    
    # Wait for API
    sleep 5
    curl -s http://localhost:8000/api/health > /dev/null || {
        echo -e "${RED}API failed to start${NC}"
        kill $API_PID 2>/dev/null
        exit 1
    }
    
    echo "Starting Worker..."
    DATABASE_URL="postgresql://coppermind:localdev@localhost:5432/coppermind" \
    REDIS_URL="redis://localhost:6379/0" \
    python -m arq worker.runner.WorkerSettings &
    WORKER_PID=$!
    
    sleep 3
    
    echo -e "${GREEN}API (PID: $API_PID) and Worker (PID: $WORKER_PID) started${NC}"
    
    cd ..
}

# Trigger pipeline
trigger_pipeline() {
    echo -e "\n${YELLOW}[6/7] Triggering pipeline...${NC}"
    
    RESPONSE=$(curl -s -X POST "http://localhost:8000/api/pipeline/trigger" \
        -H "Authorization: Bearer local-test-secret")
    
    echo "Response: $RESPONSE"
    
    # Wait for pipeline to complete
    echo "Waiting for pipeline to complete (30s)..."
    sleep 30
}

# Verify results
verify_results() {
    echo -e "\n${YELLOW}[7/7] Verifying results...${NC}"
    
    # Check news_raw
    RAW_COUNT=$(docker-compose exec -T postgres psql -U coppermind -d coppermind -t -c \
        "SELECT count(*) FROM news_raw;")
    echo "news_raw count: $RAW_COUNT"
    
    # Check news_processed
    PROC_COUNT=$(docker-compose exec -T postgres psql -U coppermind -d coppermind -t -c \
        "SELECT count(*) FROM news_processed;")
    echo "news_processed count: $PROC_COUNT"
    
    # Check metrics
    docker-compose exec -T postgres psql -U coppermind -d coppermind -c "
        SELECT run_id, status, quality_state, 
               news_raw_inserted, news_processed_inserted, 
               news_cutoff_time
        FROM pipeline_run_metrics 
        ORDER BY run_started_at DESC 
        LIMIT 1;
    "
    
    # Check API still works
    ANALYSIS=$(curl -s http://localhost:8000/api/analysis | jq -r '.quality_state // "error"')
    echo "API analysis quality_state: $ANALYSIS"
    
    if [[ "$ANALYSIS" == "error" ]]; then
        echo -e "${RED}API analysis failed${NC}"
    else
        echo -e "${GREEN}API analysis OK${NC}"
    fi
    
    echo -e "\n${GREEN}=== E2E Test Complete ===${NC}"
}

# Cleanup
cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"
    kill $API_PID 2>/dev/null || true
    kill $WORKER_PID 2>/dev/null || true
}

trap cleanup EXIT

# Main
check_prereqs
start_services
run_migration
verify_migration
# start_app  # Manual for now
# trigger_pipeline
# verify_results

echo -e "\n${GREEN}Infrastructure ready!${NC}"
echo "Next steps (manual):"
echo "  1. cd backend"
echo "  2. export DATABASE_URL=postgresql://coppermind:localdev@localhost:5432/coppermind"
echo "  3. export REDIS_URL=redis://localhost:6379/0"
echo "  4. export PIPELINE_TRIGGER_SECRET=local-test-secret"
echo "  5. uvicorn app.main:app --reload"
echo "  6. (new terminal) python -m arq worker.runner.WorkerSettings"
echo "  7. curl -X POST http://localhost:8000/api/pipeline/trigger -H 'Authorization: Bearer local-test-secret'"

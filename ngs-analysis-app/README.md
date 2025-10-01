# NGS Analysis Pipeline - Full-Stack Application

A modern full-stack application for protein selection simulation and NGS analysis, built with Next.js (TypeScript) frontend and FastAPI backend.

## Features

- **Library Generation**: Create variant libraries from multiple DNA backbones with custom degeneracy
- **Selection Simulation**: Simulate selection pressure with configurable parameters
- **NGS Simulation**: Generate next-generation sequencing reads with realistic error rates
- **Enrichment Analysis**: Analyze and visualize sequence enrichment patterns
- **Interactive UI**: Modern, responsive interface with real-time parameter updates
- **Data Visualization**: Rich charts and graphs for results analysis

## Architecture

- **Frontend**: Next.js 14 with TypeScript, Tailwind CSS, and Recharts
- **Backend**: FastAPI with Python 3.11
- **Package Integration**: Uses the `ai-ngs-analysis` package (seqlib namespace)
- **Data Flow**: RESTful API communication between frontend and backend

## Quick Start

### Prerequisites

- Node.js 18+ and npm
- Python 3.11+
- The `ai-ngs-analysis` package installed

### Development Setup

1. **Clone and navigate to the project**:
   ```bash
   cd ngs-analysis-app
   ```

2. **Install dependencies**:
   ```bash
   npm run install:all
   ```

3. **Start the development servers**:
   ```bash
   npm run dev
   ```

   This will start:
   - Backend API at http://localhost:8000
   - Frontend at http://localhost:3000

### Using Docker

1. **Build and run with Docker Compose**:
   ```bash
   docker-compose up --build
   ```

2. **Access the application**:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

## API Endpoints

### Library Generation
- `POST /api/library/generate` - Generate variant library from backbones
- `POST /api/library/random-template` - Generate random DNA template

### Selection
- `POST /api/selection/run` - Run selection simulation

### NGS Simulation
- `POST /api/ngs/simulate` - Simulate NGS reads

### Analysis
- `POST /api/analysis/run` - Run enrichment analysis

### Health Check
- `GET /api/health` - API health status

## Configuration

### Backend Configuration
The backend automatically detects the `ai-ngs-analysis` package at:
```
/Users/aparnaanandkumar/Documents/aikium/ngs_analysis_tool/ai-ngs-analysis/src
```

### Frontend Configuration
The frontend is configured to connect to the backend API at `http://localhost:8000`.

## Project Structure

```
ngs-analysis-app/
├── backend/
│   ├── main.py              # FastAPI application
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   └── page.tsx     # Main application page
│   │   └── components/      # React components
│   │       ├── LibraryGeneration.tsx
│   │       ├── Selection.tsx
│   │       ├── NGSSimulation.tsx
│   │       ├── Analysis.tsx
│   │       └── Results.tsx
│   ├── package.json
│   └── next.config.js
├── docker-compose.yml
├── Dockerfile
└── README.md
```

## Usage

1. **Library Generation**: Configure DNA backbones, weights, and degeneracy positions
2. **Selection**: Set selection parameters (rounds, stringency, target concentration, etc.)
3. **NGS Simulation**: Configure read parameters (length, error rate, abundance model)
4. **Analysis**: Run enrichment analysis and view top enriched variants
5. **Results**: View comprehensive results with visualizations and download options

## Development

### Backend Development
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Development
```bash
cd frontend
npm run dev
```

### API Documentation
Visit http://localhost:8000/docs for interactive API documentation.

## Deployment

### Production Build
```bash
npm run build
```

### Docker Deployment
```bash
docker-compose -f docker-compose.prod.yml up -d
```

## Troubleshooting

### Common Issues

1. **Package Import Errors**: Ensure the `ai-ngs-analysis` package is installed and the path is correct in `main.py`
2. **CORS Issues**: Check that the frontend URL is allowed in the CORS middleware
3. **Port Conflicts**: Ensure ports 3000 and 8000 are available

### Logs
- Backend logs: Check the terminal running the FastAPI server
- Frontend logs: Check the browser console and terminal running Next.js

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.

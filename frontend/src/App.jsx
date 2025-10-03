import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Dashboard from './components/Dashboard';
import SimulationDetails from './components/SimulationDetails';
import NewSimulation from './components/NewSimulation';
import ComparisonView from './components/ComparisonView';
import ErrorBoundary from './components/ErrorBoundary';
import ThemeToggle from './components/ThemeToggle';
import './App.css';

function App() {
  return (
    <ErrorBoundary>
      <Router>
    <ThemeToggle />
        <div className="container mt-4">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/simulations/new" element={<NewSimulation />} />
            <Route path="/simulations/:simulationId" element={<SimulationDetails />} />
            <Route path="/compare" element={<ComparisonView />} />
          </Routes>
        </div>
      </Router>
    </ErrorBoundary>
  );
}

export default App;
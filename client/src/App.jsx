import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, useLocation } from 'react-router-dom';
import Navbar from './components/Navbar';
import Sidebar from './components/Sidebar';
import Footer from './components/Footer';

// Pages
import Home from './pages/Home';
import Simulation from './pages/Simulation';
import ExperimentAnalysis from './pages/ExperimentAnalysis';
import Dashboard from './pages/Dashboard';
import History from './pages/History';
import Analytics from './pages/Analytics';
import About from './pages/About';
import NotFound from './pages/NotFound';

/**
 * Layout wrapper that renders Navbar, optional Dashboard Sidebar, and Footer
 */
const AppLayout = ({ children }) => {
  const location = useLocation();
  const isHomePage = location.pathname === '/';

  return (
    <div className="app-root-layout">
      <Navbar />

      <div className={`app-main-wrapper ${!isHomePage ? 'has-sidebar' : ''}`}>
        {!isHomePage && <Sidebar />}
        <div className="app-content-column">
          <main className="app-content-body">
            {children}
          </main>
          <Footer />
        </div>
      </div>
    </div>
  );
};

export const App = () => {
  return (
    <Router>
      <AppLayout>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/simulation" element={<Simulation />} />
          <Route path="/predict" element={<Navigate to="/simulation" replace />} />
          <Route path="/experiments" element={<ExperimentAnalysis />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/history" element={<History />} />
          <Route path="/about" element={<About />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </AppLayout>
    </Router>
  );
};

export default App;

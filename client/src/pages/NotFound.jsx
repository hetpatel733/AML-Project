import React from 'react';
import { Link } from 'react-router-dom';
import { AlertOctagon, Home, Search, LayoutDashboard } from 'lucide-react';

export const NotFound = () => {
  return (
    <div className="page-container not-found-page">
      <div className="card not-found-card text-center">
        <div className="not-found-icon-wrapper">
          <AlertOctagon size={64} className="text-danger" />
        </div>
        <span className="not-found-code font-mono">404</span>
        <h2 className="not-found-title">Page Not Found</h2>
        <p className="not-found-desc">
          The requested page route could not be found or has been relocated within the research application.
        </p>

        <div className="not-found-actions">
          <Link to="/" className="btn-primary">
            <Home size={16} />
            <span>Return to Home</span>
          </Link>
          <Link to="/simulation" className="btn-secondary">
            <Search size={16} />
            <span>Simulation Lab</span>
          </Link>
          <Link to="/dashboard" className="btn-secondary">
            <LayoutDashboard size={16} />
            <span>View Dashboard</span>
          </Link>
        </div>
      </div>
    </div>
  );
};

export default NotFound;

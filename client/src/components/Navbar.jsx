import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { 
  ShieldCheck, 
  BrainCircuit, 
  LayoutDashboard, 
  Search, 
  History as HistoryIcon, 
  BarChart3, 
  Info, 
  Menu, 
  X,
  Sparkles,
  FlaskConical,
  BookOpen
} from 'lucide-react';

export const Navbar = () => {
  const location = useLocation();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const navLinks = [
    { path: '/', label: 'Home', icon: BrainCircuit },
    { path: '/simulation', label: 'Simulation Lab', icon: FlaskConical },
    { path: '/experiments', label: 'Benchmarks', icon: BookOpen },
    { path: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { path: '/analytics', label: 'Analytics', icon: BarChart3 },
    { path: '/history', label: 'History', icon: HistoryIcon },
    { path: '/about', label: 'About', icon: Info },
  ];

  const isActive = (path) => {
    if (path === '/') return location.pathname === '/';
    return location.pathname.startsWith(path);
  };

  return (
    <header className="navbar-header">
      <div className="navbar-container">
        {/* Brand / Logo */}
        <Link to="/" className="brand-logo" onClick={() => setMobileMenuOpen(false)}>
          <div className="brand-icon-wrapper">
            <ShieldCheck className="brand-icon" size={24} />
          </div>
          <div className="brand-text-wrapper">
            <span className="brand-title">VeriNews <span className="brand-badge">AI</span></span>
            <span className="brand-subtitle">NLP & ML Fake News Classifier</span>
          </div>
        </Link>

        {/* Desktop Navigation Links */}
        <nav className="desktop-nav">
          {navLinks.map((link) => {
            const Icon = link.icon;
            const active = isActive(link.path);
            return (
              <Link
                key={link.path}
                to={link.path}
                className={`nav-link ${active ? 'nav-link-active' : ''}`}
              >
                <Icon size={16} className="nav-link-icon" />
                <span>{link.label}</span>
              </Link>
            );
          })}
        </nav>

        {/* Action Button & Status */}
        <div className="navbar-actions">
          <Link to="/simulation" className="btn-primary btn-sm">
            <FlaskConical size={15} />
            <span>Simulation Lab</span>
          </Link>

          {/* Mobile Menu Toggle Button */}
          <button 
            className="mobile-menu-btn"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            aria-label="Toggle Navigation Menu"
          >
            {mobileMenuOpen ? <X size={22} /> : <Menu size={22} />}
          </button>
        </div>
      </div>

      {/* Mobile Menu Dropdown */}
      {mobileMenuOpen && (
        <div className="mobile-nav-drawer">
          <div className="mobile-nav-links">
            {navLinks.map((link) => {
              const Icon = link.icon;
              const active = isActive(link.path);
              return (
                <Link
                  key={link.path}
                  to={link.path}
                  className={`mobile-nav-link ${active ? 'mobile-nav-link-active' : ''}`}
                  onClick={() => setMobileMenuOpen(false)}
                >
                  <Icon size={18} />
                  <span>{link.label}</span>
                </Link>
              );
            })}
          </div>
        </div>
      )}
    </header>
  );
};

export default Navbar;

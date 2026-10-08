import React from 'react';
import { NavLink } from 'react-router-dom';
import { 
  LayoutDashboard, 
  Search, 
  History, 
  BarChart3, 
  Info, 
  Sparkles,
  Layers,
  Database,
  FlaskConical,
  BookOpen
} from 'lucide-react';

export const Sidebar = () => {
  const menuItems = [
    {
      heading: 'CORE FEATURES',
      items: [
        { path: '/simulation', label: 'Simulation Lab', icon: FlaskConical, badge: 'Live Lab' },
        { path: '/dashboard', label: 'Dashboard Overview', icon: LayoutDashboard },
        { path: '/analytics', label: 'Model Analytics', icon: BarChart3 },
        { path: '/history', label: 'Prediction Archive', icon: History }
      ]
    },
    {
      heading: 'ACADEMIC RESOURCES',
      items: [
        { path: '/experiments', label: 'Benchmark Matrix', icon: BookOpen, badge: '80/20' },
        { path: '/about', label: 'NLP Architecture', icon: Info },
      ]
    }
  ];

  return (
    <aside className="app-sidebar">
      <div className="sidebar-inner">
        {menuItems.map((group, groupIdx) => (
          <div key={groupIdx} className="sidebar-group">
            <div className="sidebar-group-title">{group.heading}</div>
            <ul className="sidebar-list">
              {group.items.map((item) => {
                const Icon = item.icon;
                return (
                  <li key={item.path}>
                    <NavLink
                      to={item.path}
                      className={({ isActive }) =>
                        `sidebar-link ${isActive ? 'sidebar-link-active' : ''}`
                      }
                    >
                      <Icon size={18} className="sidebar-link-icon" />
                      <span className="sidebar-link-text">{item.label}</span>
                      {item.badge && (
                        <span className="sidebar-badge">{item.badge}</span>
                      )}
                    </NavLink>
                  </li>
                );
              })}
            </ul>
          </div>
        ))}

        {/* Academic Project Metadata in Sidebar */}
        <div className="sidebar-footer-card">
          <div className="sidebar-card-header">
            <Layers size={16} className="text-primary" />
            <span className="sidebar-card-title">NLP Pipeline</span>
          </div>
          <p className="sidebar-card-desc">
            TF-IDF (1,2-grams) with 5 classical supervised ML classifiers.
          </p>
          <div className="sidebar-api-status">
            <span className="status-dot"></span>
            <span className="status-text">Backend Ready</span>
          </div>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;

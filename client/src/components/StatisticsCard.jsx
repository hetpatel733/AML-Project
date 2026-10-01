import React from 'react';
import { TrendingUp, TrendingDown, ArrowRight } from 'lucide-react';

export const StatisticsCard = ({
  title,
  value,
  subtitle,
  icon: Icon,
  trend,
  trendDirection = 'up', // 'up' | 'down' | 'neutral'
  color = 'primary', // 'primary' | 'danger' | 'success' | 'amber'
  badge
}) => {
  const getColorClasses = () => {
    switch (color) {
      case 'danger':
        return {
          iconBg: 'rgba(239, 68, 68, 0.12)',
          iconColor: '#ef4444',
          accentBorder: '#ef4444'
        };
      case 'success':
        return {
          iconBg: 'rgba(34, 197, 94, 0.12)',
          iconColor: '#22c55e',
          accentBorder: '#22c55e'
        };
      case 'amber':
        return {
          iconBg: 'rgba(245, 158, 11, 0.12)',
          iconColor: '#f59e0b',
          accentBorder: '#f59e0b'
        };
      case 'primary':
      default:
        return {
          iconBg: 'rgba(59, 130, 246, 0.12)',
          iconColor: '#3b82f6',
          accentBorder: '#3b82f6'
        };
    }
  };

  const styleMeta = getColorClasses();

  return (
    <div className="card stats-card">
      <div className="stats-card-header">
        <div className="stats-card-title-group">
          <span className="stats-card-title">{title}</span>
          {badge && <span className="stats-card-badge">{badge}</span>}
        </div>
        {Icon && (
          <div 
            className="stats-icon-box"
            style={{ backgroundColor: styleMeta.iconBg, color: styleMeta.iconColor }}
          >
            <Icon size={22} />
          </div>
        )}
      </div>

      <div className="stats-card-body">
        <div className="stats-value-row">
          <span className="stats-card-value font-mono">{value}</span>
        </div>

        {(subtitle || trend) && (
          <div className="stats-card-footer">
            {trend && (
              <span className={`stats-trend ${trendDirection === 'up' ? 'trend-up' : 'trend-down'}`}>
                {trendDirection === 'up' ? <TrendingUp size={13} /> : <TrendingDown size={13} />}
                <span>{trend}</span>
              </span>
            )}
            {subtitle && <span className="stats-subtitle">{subtitle}</span>}
          </div>
        )}
      </div>

      {/* Subtle bottom decorative bar */}
      <div 
        className="stats-card-accent-bar" 
        style={{ backgroundColor: styleMeta.accentBorder }}
      />
    </div>
  );
};

export default StatisticsCard;

import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

export const PredictionDistribution = ({ fakeCount = 0, realCount = 0, height = 280 }) => {
  const containerRef = useRef(null);
  const svgRef = useRef(null);
  const tooltipRef = useRef(null);

  const total = fakeCount + realCount;

  useEffect(() => {
    if (!svgRef.current || !containerRef.current) return;

    // Clear previous elements
    d3.select(svgRef.current).selectAll('*').remove();

    const containerWidth = containerRef.current.clientWidth || 320;
    const width = containerWidth;
    const radius = Math.min(width, height) / 2 - 25;

    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height)
      .attr('viewBox', `0 0 ${width} ${height}`)
      .append('g')
      .attr('transform', `translate(${width / 2}, ${height / 2})`);

    // Handle empty state gracefully
    if (total === 0) {
      svg.append('circle')
        .attr('r', radius)
        .attr('fill', 'none')
        .attr('stroke', '#334155')
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', '4,4');

      svg.append('text')
        .attr('text-anchor', 'middle')
        .attr('dy', '0.3em')
        .attr('fill', '#94a3b8')
        .attr('font-size', '13px')
        .text('No Data Recorded');
      return;
    }

    const data = [
      { label: 'Real News', value: realCount, color: '#22c55e', key: 'REAL' },
      { label: 'Fake News', value: fakeCount, color: '#ef4444', key: 'FAKE' }
    ].filter(d => d.value > 0);

    const pie = d3.pie()
      .value(d => d.value)
      .sort(null)
      .padAngle(0.04);

    const arc = d3.arc()
      .innerRadius(radius * 0.58)
      .outerRadius(radius)
      .cornerRadius(6);

    const arcHover = d3.arc()
      .innerRadius(radius * 0.58)
      .outerRadius(radius + 8)
      .cornerRadius(8);

    // Create Tooltip selection
    const tooltip = d3.select(tooltipRef.current);

    // Arcs
    const paths = svg.selectAll('path')
      .data(pie(data))
      .enter()
      .append('path')
      .attr('d', arc)
      .attr('fill', d => d.data.color)
      .attr('stroke', '#0f172a')
      .attr('stroke-width', 2)
      .style('cursor', 'pointer')
      .style('transition', 'all 0.2s ease');

    // Hover interactions
    paths
      .on('mouseenter', function (event, d) {
        d3.select(this)
          .transition()
          .duration(150)
          .attr('d', arcHover)
          .attr('filter', 'drop-shadow(0 0 8px ' + d.data.color + '80)');

        const percentage = ((d.data.value / total) * 100).toFixed(1);

        tooltip
          .style('opacity', 1)
          .html(`
            <div class="d3-tooltip-title" style="color: ${d.data.color}; font-weight: 700;">
              ${d.data.label}
            </div>
            <div class="d3-tooltip-value">Count: <strong>${d.data.value}</strong></div>
            <div class="d3-tooltip-value">Ratio: <strong>${percentage}%</strong></div>
          `);
      })
      .on('mousemove', function (event) {
        const bounds = containerRef.current.getBoundingClientRect();
        tooltip
          .style('left', `${event.clientX - bounds.left + 15}px`)
          .style('top', `${event.clientY - bounds.top - 10}px`);
      })
      .on('mouseleave', function (event, d) {
        d3.select(this)
          .transition()
          .duration(150)
          .attr('d', arc)
          .attr('filter', null);

        tooltip.style('opacity', 0);
      });

    // Center text
    const centerGroup = svg.append('g').attr('class', 'donut-center-group');
    
    centerGroup.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '-0.2em')
      .attr('font-size', '22px')
      .attr('font-weight', '700')
      .attr('fill', '#f8fafc')
      .text(total);

    centerGroup.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '1.3em')
      .attr('font-size', '11px')
      .attr('font-weight', '500')
      .attr('fill', '#94a3b8')
      .text('TOTAL ARTICLES');

  }, [fakeCount, realCount, height, total]);

  return (
    <div className="d3-chart-container" ref={containerRef} style={{ position: 'relative', width: '100%' }}>
      <svg ref={svgRef} className="d3-svg-responsive" />
      <div ref={tooltipRef} className="d3-floating-tooltip" style={{ opacity: 0 }} />

      {/* Chart Legend */}
      <div className="d3-chart-legend">
        <div className="legend-item">
          <span className="legend-dot" style={{ backgroundColor: '#22c55e' }}></span>
          <span className="legend-text">Real News: <strong>{realCount}</strong> ({total > 0 ? ((realCount / total) * 100).toFixed(1) : 0}%)</span>
        </div>
        <div className="legend-item">
          <span className="legend-dot" style={{ backgroundColor: '#ef4444' }}></span>
          <span className="legend-text">Fake News: <strong>{fakeCount}</strong> ({total > 0 ? ((fakeCount / total) * 100).toFixed(1) : 0}%)</span>
        </div>
      </div>
    </div>
  );
};

export default PredictionDistribution;

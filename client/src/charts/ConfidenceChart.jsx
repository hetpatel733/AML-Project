import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

export const ConfidenceChart = ({ data = [], height = 280 }) => {
  const containerRef = useRef(null);
  const svgRef = useRef(null);
  const tooltipRef = useRef(null);

  useEffect(() => {
    if (!svgRef.current || !containerRef.current) return;

    d3.select(svgRef.current).selectAll('*').remove();

    const containerWidth = containerRef.current.clientWidth || 400;
    const margin = { top: 25, right: 20, bottom: 45, left: 45 };
    const width = containerWidth - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;

    const svg = d3.select(svgRef.current)
      .attr('width', containerWidth)
      .attr('height', height)
      .attr('viewBox', `0 0 ${containerWidth} ${height}`)
      .append('g')
      .attr('transform', `translate(${margin.left}, ${margin.top})`);

    const chartData = (data && data.length > 0) ? data : [
      { range: '50-60%', count: 0 },
      { range: '60-70%', count: 0 },
      { range: '70-80%', count: 0 },
      { range: '80-90%', count: 0 },
      { range: '90-100%', count: 0 }
    ];

    const maxVal = d3.max(chartData, d => d.count) || 10;
    const totalCount = d3.sum(chartData, d => d.count);

    // X scale
    const x = d3.scaleBand()
      .domain(chartData.map(d => d.range))
      .range([0, width])
      .padding(0.3);

    // Y scale
    const y = d3.scaleLinear()
      .domain([0, Math.ceil(maxVal * 1.15)])
      .nice()
      .range([chartHeight, 0]);

    // Background horizontal grid lines
    svg.append('g')
      .attr('class', 'grid-lines')
      .call(
        d3.axisLeft(y)
          .tickSize(-width)
          .tickFormat('')
          .ticks(5)
      )
      .selectAll('line')
      .attr('stroke', 'rgba(255, 255, 255, 0.06)')
      .attr('stroke-dasharray', '3,3');

    // Gradient definitions
    const defs = svg.append('defs');
    const gradient = defs.append('linearGradient')
      .attr('id', 'conf-bar-gradient')
      .attr('x1', '0%').attr('y1', '0%')
      .attr('x2', '0%').attr('y2', '100%');

    gradient.append('stop')
      .attr('offset', '0%')
      .attr('stop-color', '#3b82f6');

    gradient.append('stop')
      .attr('offset', '100%')
      .attr('stop-color', '#1d4ed8');

    // Tooltip
    const tooltip = d3.select(tooltipRef.current);

    // Bars
    svg.selectAll('.bar')
      .data(chartData)
      .enter()
      .append('rect')
      .attr('class', 'bar')
      .attr('x', d => x(d.range))
      .attr('y', d => y(d.count))
      .attr('width', x.bandwidth())
      .attr('height', d => Math.max(0, chartHeight - y(d.count)))
      .attr('rx', 4)
      .attr('fill', 'url(#conf-bar-gradient)')
      .attr('stroke', '#60a5fa')
      .attr('stroke-width', 1)
      .style('cursor', 'pointer')
      .style('transition', 'all 0.2s ease')
      .on('mouseenter', function (event, d) {
        d3.select(this)
          .attr('fill', '#60a5fa')
          .attr('filter', 'drop-shadow(0 0 8px rgba(96, 165, 250, 0.6))');

        const pct = totalCount > 0 ? ((d.count / totalCount) * 100).toFixed(1) : 0;

        tooltip
          .style('opacity', 1)
          .html(`
            <div class="d3-tooltip-title" style="color: #60a5fa; font-weight: 700;">Confidence Range: ${d.range}</div>
            <div class="d3-tooltip-value">Predictions: <strong>${d.count}</strong></div>
            <div class="d3-tooltip-value">Percentage: <strong>${pct}%</strong></div>
          `);
      })
      .on('mousemove', function (event) {
        const bounds = containerRef.current.getBoundingClientRect();
        tooltip
          .style('left', `${event.clientX - bounds.left + 15}px`)
          .style('top', `${event.clientY - bounds.top - 10}px`);
      })
      .on('mouseleave', function () {
        d3.select(this)
          .attr('fill', 'url(#conf-bar-gradient)')
          .attr('filter', null);
        tooltip.style('opacity', 0);
      });

    // Value Labels on Top of Bars
    svg.selectAll('.bar-label')
      .data(chartData)
      .enter()
      .append('text')
      .attr('class', 'bar-label font-mono')
      .attr('x', d => x(d.range) + x.bandwidth() / 2)
      .attr('y', d => y(d.count) - 6)
      .attr('text-anchor', 'middle')
      .attr('fill', '#94a3b8')
      .attr('font-size', '11px')
      .text(d => d.count > 0 ? d.count : '');

    // X Axis
    svg.append('g')
      .attr('transform', `translate(0, ${chartHeight})`)
      .call(d3.axisBottom(x))
      .attr('color', '#64748b')
      .selectAll('text')
      .attr('fill', '#94a3b8')
      .attr('font-size', '11px')
      .attr('dy', '1em');

    // Y Axis
    svg.append('g')
      .call(d3.axisLeft(y).ticks(5).tickFormat(d3.format('d')))
      .attr('color', '#64748b')
      .selectAll('text')
      .attr('fill', '#94a3b8')
      .attr('font-size', '11px');

    // Y Axis Label
    svg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', 0 - margin.left + 14)
      .attr('x', 0 - (chartHeight / 2))
      .attr('text-anchor', 'middle')
      .attr('fill', '#64748b')
      .attr('font-size', '11px')
      .text('Prediction Count');

  }, [data, height]);

  return (
    <div className="d3-chart-container" ref={containerRef} style={{ position: 'relative', width: '100%' }}>
      <svg ref={svgRef} className="d3-svg-responsive" />
      <div ref={tooltipRef} className="d3-floating-tooltip" style={{ opacity: 0 }} />
    </div>
  );
};

export default ConfidenceChart;

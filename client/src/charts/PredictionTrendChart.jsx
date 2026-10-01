import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

export const PredictionTrendChart = ({ data = [], height = 300 }) => {
  const containerRef = useRef(null);
  const svgRef = useRef(null);
  const tooltipRef = useRef(null);

  useEffect(() => {
    if (!svgRef.current || !containerRef.current) return;

    d3.select(svgRef.current).selectAll('*').remove();

    const containerWidth = containerRef.current.clientWidth || 500;
    const margin = { top: 20, right: 30, bottom: 40, left: 45 };
    const width = containerWidth - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;

    const svg = d3.select(svgRef.current)
      .attr('width', containerWidth)
      .attr('height', height)
      .attr('viewBox', `0 0 ${containerWidth} ${height}`)
      .append('g')
      .attr('transform', `translate(${margin.left}, ${margin.top})`);

    const chartData = (data && data.length > 0) ? data : [
      { date: 'Day 1', fake: 2, real: 4, total: 6 },
      { date: 'Day 2', fake: 3, real: 5, total: 8 },
      { date: 'Day 3', fake: 1, real: 6, total: 7 },
      { date: 'Day 4', fake: 4, real: 4, total: 8 },
      { date: 'Day 5', fake: 5, real: 7, total: 12 },
      { date: 'Day 6', fake: 3, real: 8, total: 11 },
      { date: 'Day 7', fake: 6, real: 9, total: 15 }
    ];

    // X scale
    const x = d3.scalePoint()
      .domain(chartData.map(d => d.date))
      .range([0, width])
      .padding(0.2);

    // Y scale
    const maxVal = d3.max(chartData, d => Math.max(d.fake, d.real, d.total || 0)) || 10;
    const y = d3.scaleLinear()
      .domain([0, Math.ceil(maxVal * 1.2)])
      .nice()
      .range([chartHeight, 0]);

    // Grid lines
    svg.append('g')
      .attr('class', 'grid-lines')
      .call(
        d3.axisLeft(y)
          .tickSize(-width)
          .tickFormat('')
          .ticks(5)
      )
      .selectAll('line')
      .attr('stroke', 'rgba(255, 255, 255, 0.05)')
      .attr('stroke-dasharray', '3,3');

    // Gradients
    const defs = svg.append('defs');

    // Real gradient
    const gradReal = defs.append('linearGradient')
      .attr('id', 'area-gradient-real')
      .attr('x1', '0%').attr('y1', '0%')
      .attr('x2', '0%').attr('y2', '100%');
    gradReal.append('stop').attr('offset', '0%').attr('stop-color', '#22c55e').attr('stop-opacity', 0.25);
    gradReal.append('stop').attr('offset', '100%').attr('stop-color', '#22c55e').attr('stop-opacity', 0.0);

    // Fake gradient
    const gradFake = defs.append('linearGradient')
      .attr('id', 'area-gradient-fake')
      .attr('x1', '0%').attr('y1', '0%')
      .attr('x2', '0%').attr('y2', '100%');
    gradFake.append('stop').attr('offset', '0%').attr('stop-color', '#ef4444').attr('stop-opacity', 0.25);
    gradFake.append('stop').attr('offset', '100%').attr('stop-color', '#ef4444').attr('stop-opacity', 0.0);

    // Area generators
    const areaReal = d3.area()
      .x(d => x(d.date))
      .y0(chartHeight)
      .y1(d => y(d.real))
      .curve(d3.curveMonotoneX);

    const areaFake = d3.area()
      .x(d => x(d.date))
      .y0(chartHeight)
      .y1(d => y(d.fake))
      .curve(d3.curveMonotoneX);

    // Line generators
    const lineReal = d3.line()
      .x(d => x(d.date))
      .y(d => y(d.real))
      .curve(d3.curveMonotoneX);

    const lineFake = d3.line()
      .x(d => x(d.date))
      .y(d => y(d.fake))
      .curve(d3.curveMonotoneX);

    // Draw Areas
    svg.append('path')
      .datum(chartData)
      .attr('fill', 'url(#area-gradient-real)')
      .attr('d', areaReal);

    svg.append('path')
      .datum(chartData)
      .attr('fill', 'url(#area-gradient-fake)')
      .attr('d', areaFake);

    // Draw Lines
    svg.append('path')
      .datum(chartData)
      .attr('fill', 'none')
      .attr('stroke', '#22c55e')
      .attr('stroke-width', 2.5)
      .attr('d', lineReal);

    svg.append('path')
      .datum(chartData)
      .attr('fill', 'none')
      .attr('stroke', '#ef4444')
      .attr('stroke-width', 2.5)
      .attr('d', lineFake);

    // Tooltip
    const tooltip = d3.select(tooltipRef.current);

    // Interactive Overlay & Points
    chartData.forEach(d => {
      // Real Point
      svg.append('circle')
        .attr('cx', x(d.date))
        .attr('cy', y(d.real))
        .attr('r', 4.5)
        .attr('fill', '#22c55e')
        .attr('stroke', '#0f172a')
        .attr('stroke-width', 2);

      // Fake Point
      svg.append('circle')
        .attr('cx', x(d.date))
        .attr('cy', y(d.fake))
        .attr('r', 4.5)
        .attr('fill', '#ef4444')
        .attr('stroke', '#0f172a')
        .attr('stroke-width', 2);
    });

    // Crosshair line
    const crosshair = svg.append('line')
      .attr('class', 'crosshair-line')
      .attr('y1', 0)
      .attr('y2', chartHeight)
      .attr('stroke', '#64748b')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '3,3')
      .style('opacity', 0);

    // Hover rect overlay
    svg.append('rect')
      .attr('width', width)
      .attr('height', chartHeight)
      .attr('fill', 'transparent')
      .style('cursor', 'crosshair')
      .on('mousemove', function (event) {
        const [mouseX] = d3.pointer(event);
        
        // Find nearest date point
        let nearest = chartData[0];
        let minDist = Infinity;
        chartData.forEach(d => {
          const dist = Math.abs(x(d.date) - mouseX);
          if (dist < minDist) {
            minDist = dist;
            nearest = d;
          }
        });

        const posX = x(nearest.date);
        crosshair
          .attr('x1', posX)
          .attr('x2', posX)
          .style('opacity', 1);

        const bounds = containerRef.current.getBoundingClientRect();
        tooltip
          .style('opacity', 1)
          .style('left', `${event.clientX - bounds.left + 15}px`)
          .style('top', `${event.clientY - bounds.top - 10}px`)
          .html(`
            <div class="d3-tooltip-title" style="font-weight: 700;">${nearest.date}</div>
            <div class="d3-tooltip-value" style="color: #22c55e;">Real Verified: <strong>${nearest.real}</strong></div>
            <div class="d3-tooltip-value" style="color: #ef4444;">Fake Detected: <strong>${nearest.fake}</strong></div>
            <div class="d3-tooltip-value" style="color: #94a3b8; border-top: 1px solid #334155; margin-top: 4px; padding-top: 4px;">
              Total: <strong>${nearest.real + nearest.fake}</strong>
            </div>
          `);
      })
      .on('mouseleave', function () {
        crosshair.style('opacity', 0);
        tooltip.style('opacity', 0);
      });

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
      .text('Daily Count');

  }, [data, height]);

  return (
    <div className="d3-chart-container" ref={containerRef} style={{ position: 'relative', width: '100%' }}>
      <svg ref={svgRef} className="d3-svg-responsive" />
      <div ref={tooltipRef} className="d3-floating-tooltip" style={{ opacity: 0 }} />

      {/* Legend */}
      <div className="d3-chart-legend">
        <div className="legend-item">
          <span className="legend-line" style={{ backgroundColor: '#22c55e' }}></span>
          <span className="legend-text">Real News</span>
        </div>
        <div className="legend-item">
          <span className="legend-line" style={{ backgroundColor: '#ef4444' }}></span>
          <span className="legend-text">Fake News</span>
        </div>
      </div>
    </div>
  );
};

export default PredictionTrendChart;

import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';

export const ModelPerformanceChart = ({ data = [], height = 340 }) => {
  const containerRef = useRef(null);
  const svgRef = useRef(null);
  const tooltipRef = useRef(null);

  const metrics = [
    { key: 'accuracy', label: 'Accuracy', color: '#3b82f6' },
    { key: 'precision', label: 'Precision', color: '#10b981' },
    { key: 'recall', label: 'Recall', color: '#8b5cf6' },
    { key: 'f1Score', label: 'F1 Score', color: '#f59e0b' }
  ];

  useEffect(() => {
    if (!svgRef.current || !containerRef.current) return;

    d3.select(svgRef.current).selectAll('*').remove();

    const containerWidth = containerRef.current.clientWidth || 550;
    const margin = { top: 25, right: 20, bottom: 50, left: 45 };
    const width = containerWidth - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;

    const svg = d3.select(svgRef.current)
      .attr('width', containerWidth)
      .attr('height', height)
      .attr('viewBox', `0 0 ${containerWidth} ${height}`)
      .append('g')
      .attr('transform', `translate(${margin.left}, ${margin.top})`);

    const chartData = (data && data.length > 0) ? data : [
      { model: 'Passive Aggressive', accuracy: 0.948, precision: 0.952, recall: 0.941, f1Score: 0.946 },
      { model: 'Logistic Regression', accuracy: 0.936, precision: 0.931, recall: 0.942, f1Score: 0.936 },
      { model: 'Linear SVM', accuracy: 0.941, precision: 0.945, recall: 0.935, f1Score: 0.940 },
      { model: 'Multinomial NB', accuracy: 0.894, precision: 0.887, recall: 0.902, f1Score: 0.894 },
      { model: 'Random Forest', accuracy: 0.912, precision: 0.920, recall: 0.901, f1Score: 0.910 }
    ];

    // Main X scale (models)
    const x0 = d3.scaleBand()
      .domain(chartData.map(d => d.model))
      .range([0, width])
      .padding(0.25);

    // Sub X scale (metrics per model)
    const x1 = d3.scaleBand()
      .domain(metrics.map(m => m.key))
      .range([0, x0.bandwidth()])
      .padding(0.08);

    // Y scale (0 to 1.0 or 0% to 100%)
    const y = d3.scaleLinear()
      .domain([0.7, 1.0])
      .range([chartHeight, 0]);

    // Grid lines
    svg.append('g')
      .attr('class', 'grid-lines')
      .call(
        d3.axisLeft(y)
          .tickSize(-width)
          .tickFormat('')
          .ticks(6)
      )
      .selectAll('line')
      .attr('stroke', 'rgba(255, 255, 255, 0.05)')
      .attr('stroke-dasharray', '3,3');

    // Tooltip
    const tooltip = d3.select(tooltipRef.current);

    // Render grouped bars
    const modelGroup = svg.selectAll('.model-group')
      .data(chartData)
      .enter()
      .append('g')
      .attr('class', 'model-group')
      .attr('transform', d => `translate(${x0(d.model)}, 0)`);

    metrics.forEach(metric => {
      modelGroup.append('rect')
        .attr('class', 'metric-bar')
        .attr('x', x1(metric.key))
        .attr('y', d => y(d[metric.key] || 0.7))
        .attr('width', x1.bandwidth())
        .attr('height', d => Math.max(0, chartHeight - y(d[metric.key] || 0.7)))
        .attr('fill', metric.color)
        .attr('rx', 3)
        .style('cursor', 'pointer')
        .style('transition', 'all 0.2s ease')
        .on('mouseenter', function (event, d) {
          d3.select(this)
            .attr('filter', `drop-shadow(0 0 6px ${metric.color}90)`);

          const val = ((d[metric.key] || 0) * 100).toFixed(1);

          tooltip
            .style('opacity', 1)
            .html(`
              <div class="d3-tooltip-title" style="font-weight: 700;">${d.model}</div>
              <div class="d3-tooltip-value" style="color: ${metric.color};">
                ${metric.label}: <strong>${val}%</strong>
              </div>
            `);
        })
        .on('mousemove', function (event) {
          const bounds = containerRef.current.getBoundingClientRect();
          tooltip
            .style('left', `${event.clientX - bounds.left + 15}px`)
            .style('top', `${event.clientY - bounds.top - 10}px`);
        })
        .on('mouseleave', function () {
          d3.select(this).attr('filter', null);
          tooltip.style('opacity', 0);
        });
    });

    // X Axis
    svg.append('g')
      .attr('transform', `translate(0, ${chartHeight})`)
      .call(d3.axisBottom(x0))
      .attr('color', '#64748b')
      .selectAll('text')
      .attr('fill', '#cbd5e1')
      .attr('font-size', '11px')
      .attr('dy', '1.2em');

    // Y Axis
    svg.append('g')
      .call(
        d3.axisLeft(y)
          .ticks(6)
          .tickFormat(d => `${(d * 100).toFixed(0)}%`)
      )
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
      .text('Evaluation Score (0 - 100%)');

  }, [data, height]);

  return (
    <div className="d3-chart-container" ref={containerRef} style={{ position: 'relative', width: '100%' }}>
      <svg ref={svgRef} className="d3-svg-responsive" />
      <div ref={tooltipRef} className="d3-floating-tooltip" style={{ opacity: 0 }} />

      {/* Legend */}
      <div className="d3-chart-legend">
        {metrics.map(m => (
          <div key={m.key} className="legend-item">
            <span className="legend-dot" style={{ backgroundColor: m.color }}></span>
            <span className="legend-text">{m.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ModelPerformanceChart;

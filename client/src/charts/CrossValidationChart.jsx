import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';

/**
 * CrossValidationChart
 * D3.js visualization showing 5-Fold Cross Validation Mean and Standard Deviation error bars.
 */
export const CrossValidationChart = ({ cvData = {}, height = 340 }) => {
  const containerRef = useRef(null);
  const svgRef = useRef(null);
  const tooltipRef = useRef(null);
  const [selectedMetric, setSelectedMetric] = useState('f1'); // 'accuracy', 'f1', 'roc_auc'

  useEffect(() => {
    if (!svgRef.current || !containerRef.current) return;

    d3.select(svgRef.current).selectAll('*').remove();

    const containerWidth = containerRef.current.clientWidth || 600;
    const margin = { top: 25, right: 30, bottom: 65, left: 60 };
    const width = containerWidth - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;

    const svg = d3.select(svgRef.current)
      .attr('width', containerWidth)
      .attr('height', height)
      .attr('viewBox', `0 0 ${containerWidth} ${height}`)
      .append('g')
      .attr('transform', `translate(${margin.left}, ${margin.top})`);

    // Normalize the authoritative benchmark array and retain legacy object support.
    let rawData = cvData?.models || cvData?.model_metrics || cvData || {};
    if (rawData.models) rawData = rawData.models;
    else if (rawData.model_metrics) rawData = rawData.model_metrics;

    const ignoredKeys = new Set([
      'dataset_metadata',
      'split_ratios',
      'split_counts',
      'ensembles',
      'vectorizers',
      'validation_weights',
      'ensemble_weights',
      'cv_folds'
    ]);

    const modelEntries = Array.isArray(rawData)
      ? rawData.map((item, index) => [item?.id || `model-${index}`, item])
      : Object.entries(rawData).filter(([key, item]) => (
        !ignoredKeys.has(key) && typeof item === 'object' && item !== null
      ));

    const items = modelEntries.map(([key, item]) => {
      if (!item || typeof item !== 'object') return null;
      if (item.status === 'unavailable') return null;

      const cvMean = item.crossValidation?.mean || {};
      const cvStd = item.crossValidation?.std || {};
      const cv = item.cv_metrics || {};
      let mean;
      let std;

      if (selectedMetric === 'f1') {
        mean = cvMean.f1 ?? cv.f1_mean ?? cv.f1 ?? item.cv_f1_mean;
        std = cvStd.f1 ?? cv.f1_std ?? item.cv_f1_std;
      } else if (selectedMetric === 'accuracy') {
        mean = cvMean.accuracy ?? cv.accuracy_mean ?? cv.accuracy ?? item.cv_accuracy_mean;
        std = cvStd.accuracy ?? cv.accuracy_std ?? item.cv_accuracy_std;
      } else if (selectedMetric === 'roc_auc') {
        mean = cvMean.rocAuc ?? cvMean.roc_auc ?? cv.roc_auc_mean ?? cv.roc_auc ?? item.cv_roc_auc_mean;
        std = cvStd.rocAuc ?? cvStd.roc_auc ?? cv.roc_auc_std ?? item.cv_roc_auc_std;
      }

      if (typeof mean !== 'number' || Number.isNaN(mean)) return null;
      std = typeof std === 'number' && !Number.isNaN(std) ? std : 0;

      const fullName = item.name || item.model_name || key;
      const name = fullName
        .replace('Multinomial Naive Bayes', 'MNB')
        .replace('Passive Aggressive Classifier', 'PAC')
        .replace('Passive Aggressive', 'PAC')
        .replace('Logistic Regression', 'LogReg')
        .replace('Random Forest', 'RF')
        .replace('Support Vector Machine', 'SVM')
        .replace('Linear SVM', 'SVM');

      return {
        key,
        name,
        fullName,
        mean,
        std,
        min: Math.max(0, mean - std),
        max: Math.min(1.0, mean + std)
      };
    }).filter(Boolean);

    if (items.length === 0) return;

    // Scales
    const x = d3.scaleBand()
      .domain(items.map(d => d.name))
      .range([0, width])
      .padding(0.35);

    // Dynamic Y domain
    const minVal = Math.min(...items.map(d => d.min));
    const yMin = Math.max(0, Math.floor((minVal - 0.05) * 20) / 20); // rounded to nearest 0.05
    const y = d3.scaleLinear()
      .domain([Math.min(yMin, 0.85), 1.0])
      .range([chartHeight, 0]);

    // Grid Lines
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

    const tooltip = d3.select(tooltipRef.current);

    // Color gradient / fill
    const colorScale = d3.scaleOrdinal()
      .domain(items.map(d => d.key))
      .range(['#3b82f6', '#06b6d4', '#8b5cf6', '#a855f7', '#f59e0b', '#f97316']);

    // Draw Bars
    svg.selectAll('.cv-bar')
      .data(items)
      .enter()
      .append('rect')
      .attr('class', 'cv-bar')
      .attr('x', d => x(d.name))
      .attr('y', d => y(d.mean))
      .attr('width', x.bandwidth())
      .attr('height', d => Math.max(0, chartHeight - y(d.mean)))
      .attr('rx', 4)
      .attr('fill', d => colorScale(d.key))
      .attr('opacity', 0.85)
      .style('cursor', 'pointer')
      .on('mouseenter', function (event, d) {
        d3.select(this)
          .attr('opacity', 1.0)
          .attr('filter', `drop-shadow(0 0 8px ${colorScale(d.key)}90)`);

        const metricLabel = selectedMetric === 'f1' ? 'F1 Score' : selectedMetric === 'accuracy' ? 'Accuracy' : 'ROC-AUC';
        tooltip
          .style('opacity', 1)
          .html(`
            <div style="font-weight: 700; color: #f8fafc;">${d.fullName}</div>
            <div style="color: ${colorScale(d.key)}; font-weight: 600; margin-top: 2px;">
              5-Fold CV Mean ${metricLabel}: ${(d.mean * 100).toFixed(2)}%
            </div>
            <div style="font-size: 11px; color: #94a3b8; margin-top: 2px;">
              Std Dev (σ): ±${(d.std * 100).toFixed(3)}%
            </div>
            <div style="font-size: 11px; color: #cbd5e1; margin-top: 2px;">
              Interval: [${(d.min * 100).toFixed(2)}%, ${(d.max * 100).toFixed(2)}%]
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
        d3.select(this).attr('opacity', 0.85).attr('filter', null);
        tooltip.style('opacity', 0);
      });

    // Draw Error Bars (Whiskers: mean ± std)
    const whiskerGroup = svg.selectAll('.whisker-group')
      .data(items)
      .enter()
      .append('g')
      .attr('class', 'whisker-group');

    // Vertical line
    whiskerGroup.append('line')
      .attr('x1', d => (x(d.name) || 0) + x.bandwidth() / 2)
      .attr('x2', d => (x(d.name) || 0) + x.bandwidth() / 2)
      .attr('y1', d => y(d.max))
      .attr('y2', d => y(d.min))
      .attr('stroke', '#ffffff')
      .attr('stroke-width', 2);

    // Top cap
    whiskerGroup.append('line')
      .attr('x1', d => (x(d.name) || 0) + x.bandwidth() / 2 - 6)
      .attr('x2', d => (x(d.name) || 0) + x.bandwidth() / 2 + 6)
      .attr('y1', d => y(d.max))
      .attr('y2', d => y(d.max))
      .attr('stroke', '#ffffff')
      .attr('stroke-width', 2);

    // Bottom cap
    whiskerGroup.append('line')
      .attr('x1', d => (x(d.name) || 0) + x.bandwidth() / 2 - 6)
      .attr('x2', d => (x(d.name) || 0) + x.bandwidth() / 2 + 6)
      .attr('y1', d => y(d.min))
      .attr('y2', d => y(d.min))
      .attr('stroke', '#ffffff')
      .attr('stroke-width', 2);

    // Mean Value Labels
    svg.selectAll('.cv-label')
      .data(items)
      .enter()
      .append('text')
      .attr('x', d => (x(d.name) || 0) + x.bandwidth() / 2)
      .attr('y', d => y(d.max) - 8)
      .attr('text-anchor', 'middle')
      .attr('fill', '#f1f5f9')
      .attr('font-size', '11px')
      .attr('font-weight', '600')
      .text(d => `${(d.mean * 100).toFixed(1)}%`);

    // X Axis
    svg.append('g')
      .attr('transform', `translate(0, ${chartHeight})`)
      .call(d3.axisBottom(x))
      .selectAll('text')
      .attr('transform', 'rotate(-25)')
      .style('text-anchor', 'end')
      .attr('dx', '-0.5em')
      .attr('dy', '0.7em')
      .attr('fill', '#94a3b8')
      .attr('font-size', '11px');

    // Y Axis
    svg.append('g')
      .call(
        d3.axisLeft(y)
          .ticks(5)
          .tickFormat(d => `${(d * 100).toFixed(0)}%`)
      )
      .selectAll('text')
      .attr('fill', '#94a3b8')
      .attr('font-size', '11px');

  }, [cvData, selectedMetric, height]);

  return (
    <div className="cv-chart-card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
        <h4 style={{ margin: 0, fontSize: '15px', color: '#f8fafc', fontWeight: 600 }}>
          5-Fold Stratified Cross-Validation (Mean ± 1σ)
        </h4>
        <div style={{ display: 'flex', gap: '8px' }}>
          {['f1', 'accuracy', 'roc_auc'].map(metric => (
            <button
              key={metric}
              type="button"
              onClick={() => setSelectedMetric(metric)}
              style={{
                background: selectedMetric === metric ? 'rgba(59, 130, 246, 0.25)' : 'rgba(255, 255, 255, 0.05)',
                border: selectedMetric === metric ? '1px solid #3b82f6' : '1px solid rgba(255, 255, 255, 0.1)',
                color: selectedMetric === metric ? '#60a5fa' : '#94a3b8',
                borderRadius: '6px',
                padding: '4px 10px',
                fontSize: '12px',
                fontWeight: 600,
                cursor: 'pointer',
                transition: 'all 0.2s ease'
              }}
            >
              {metric === 'f1' ? 'F1 Score' : metric === 'accuracy' ? 'Accuracy' : 'ROC-AUC'}
            </button>
          ))}
        </div>
      </div>

      <div className="chart-wrapper" ref={containerRef} style={{ position: 'relative', width: '100%' }}>
        <svg ref={svgRef} style={{ width: '100%', overflow: 'visible' }} />
        <div
          ref={tooltipRef}
          className="d3-chart-tooltip"
          style={{
            position: 'absolute',
            opacity: 0,
            pointerEvents: 'none',
            backgroundColor: 'rgba(15, 23, 42, 0.95)',
            border: '1px solid rgba(255, 255, 255, 0.15)',
            borderRadius: '8px',
            padding: '8px 12px',
            color: '#fff',
            fontSize: '12px',
            zIndex: 50,
            boxShadow: '0 10px 25px rgba(0,0,0,0.5)',
            backdropFilter: 'blur(8px)',
            transition: 'opacity 0.15s ease'
          }}
        />
      </div>
    </div>
  );
};

export default CrossValidationChart;

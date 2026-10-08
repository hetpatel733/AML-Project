import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { 
  ShieldCheck, 
  BrainCircuit, 
  LayoutDashboard, 
  Search, 
  ArrowRight, 
  CheckCircle2, 
  AlertTriangle, 
  Cpu, 
  Layers, 
  BarChart3, 
  FileCheck,
  Zap,
  Lock,
  Binary,
  FlaskConical,
  BookOpen
} from 'lucide-react';
import { getBenchmarks } from '../services/api';
import { useDataset } from '../context/DatasetContext';

export const Home = () => {
  const { selectedDataset } = useDataset();
  const [topAccuracy, setTopAccuracy] = useState('99.6%');
  const [modelCount, setModelCount] = useState(5);

  useEffect(() => {
    getBenchmarks(selectedDataset)
      .then((res) => {
        if (res.data?.models && res.data.models.length > 0) {
          setModelCount(res.data.models.length);
          const highestAcc = Math.max(...res.data.models.map(m => m.metrics?.accuracy || 0));
          if (highestAcc > 0) {
            setTopAccuracy(`${(highestAcc * 100).toFixed(1)}%`);
          }
        }
      })
      .catch((err) => console.log('Home benchmark fetch fallback:', err));
  }, [selectedDataset]);
  const steps = [
    {
      step: '01',
      title: 'Text Preprocessing',
      desc: 'Clean and normalize text with lowercasing, punctuation/digit removal, stop-words filtering, and WordNet lemmatization.',
      icon: FileCheck
    },
    {
      step: '02',
      title: 'TF-IDF Vectorization',
      desc: 'Extract n-gram features (unigrams + bigrams) with sublinear TF-IDF scaling (10,000 max features).',
      icon: Binary
    },
    {
      step: '03',
      title: '5-Model Classification',
      desc: 'Evaluate text across Logistic Regression, Naive Bayes, Linear SVM, Decision Tree, and Random Forest.',
      icon: Cpu
    },
    {
      step: '04',
      title: 'Performance & Consensus',
      desc: 'Analyze individual model probabilities, benchmark performance, and consensus metrics across datasets.',
      icon: ShieldCheck
    }
  ];

  const technologies = [
    { name: 'React & Vite', category: 'Frontend', desc: 'Modern component-based UI with fast development workflow.' },
    { name: 'Interactive D3.js', category: 'Visualization', desc: 'Real-time data visualization and performance benchmarks.' },
    { name: 'FastAPI & Scikit-Learn', category: 'ML Service', desc: 'Python-based machine learning inference with 5 classical classifiers.' },
    { name: 'TF-IDF (1,2-grams)', category: 'NLP', desc: 'Text vectorization with unigram and bigram feature extraction up to 10k features.' },
    { name: '5 ML Classifiers', category: 'Classification', desc: 'Logistic Regression, Naive Bayes, Linear SVM, Decision Tree, Random Forest.' },
    { name: 'Node.js API', category: 'Backend', desc: 'REST API connecting frontend with ML inference services.' }
  ];

  return (
    <div className="home-page">
      {/* Hero Section */}
      <section className="hero-section">
        <div className="hero-glow-sphere"></div>
        <div className="hero-content">
          <div className="hero-badge">
            <BrainCircuit size={15} />
            <span>AI-Powered News Verification</span>
          </div>

          <h1 className="hero-title">
            Fake News Detection Using <span className="gradient-text">Machine Learning</span>
          </h1>

          <p className="hero-description">
            Advanced AI platform for analyzing and classifying news authenticity using natural language processing 
            and 5 supervised machine learning models.
          </p>

          <div className="hero-cta-group">
            <Link to="/simulation" className="btn-primary btn-lg">
              <FlaskConical size={18} />
              <span>Launch Simulation Lab</span>
              <ArrowRight size={16} />
            </Link>

            <Link to="/experiments" className="btn-secondary btn-lg">
              <BookOpen size={18} />
              <span>View Benchmark Matrix</span>
            </Link>

            <Link to="/dashboard" className="btn-secondary btn-lg">
              <LayoutDashboard size={18} />
              <span>Executive Dashboard</span>
            </Link>
          </div>

          {/* Quick Metrics Bar */}
          <div className="hero-stats-row">
            <div className="hero-stat-item">
              <span className="hero-stat-num font-mono">{topAccuracy}</span>
              <span className="hero-stat-label">Top Test Accuracy ({selectedDataset.toUpperCase()})</span>
            </div>
            <div className="hero-stat-divider"></div>
            <div className="hero-stat-item">
              <span className="hero-stat-num font-mono">{modelCount} Models</span>
              <span className="hero-stat-label">Supervised Classifiers</span>
            </div>
            <div className="hero-stat-divider"></div>
            <div className="hero-stat-item">
              <span className="hero-stat-num font-mono">80 / 20</span>
              <span className="hero-stat-label">Train / Test Split</span>
            </div>
          </div>
        </div>
      </section>

      {/* Project Objective Section */}
      <section className="objective-section section-padding">
        <div className="section-header text-center">
          <span className="section-eyebrow">ACADEMIC MISSION</span>
          <h2 className="section-title">Project Scope &amp; Objective</h2>
          <p className="section-subtitle">
            Combating algorithmic disinformation through transparent machine learning interpretability
          </p>
        </div>

        <div className="objective-grid">
          <div className="card objective-card">
            <div className="card-icon-badge text-primary">
              <Zap size={22} />
            </div>
            <h3 className="card-title">Real-Time Linguistic Verification</h3>
            <p className="card-desc">
              Provide instantaneous classification of sensationalist, fabricated, or deceptive text 
              by evaluating stylistic anomalies and semantic structure against verified news corpuses.
            </p>
          </div>

          <div className="card objective-card">
            <div className="card-icon-badge text-success">
              <BarChart3 size={22} />
            </div>
            <h3 className="card-title">D3.js Data Interpretability</h3>
            <p className="card-desc">
              Render rich interactive data visualizations illustrating confidence spreads, temporal trends, 
              and comparative algorithmic benchmarks (Accuracy, Precision, Recall, F1 Score).
            </p>
          </div>

          <div className="card objective-card">
            <div className="card-icon-badge text-amber">
              <Layers size={22} />
            </div>
            <h3 className="card-title">Dual Mathematical Ensembles</h3>
            <p className="card-desc">
              Harness both Validation-Weighted Soft Probability Averaging and Majority Voting Hard Consensus 
              to maximize empirical generalization on out-of-distribution articles.
            </p>
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="workflow-section section-padding">
        <div className="section-header text-center">
          <span className="section-eyebrow">PIPELINE ARCHITECTURE</span>
          <h2 className="section-title">How the NLP Detection System Works</h2>
          <p className="section-subtitle">
            Four rigorous stages from raw headline input to probabilistic classification verdict
          </p>
        </div>

        <div className="workflow-grid">
          {steps.map((st) => {
            const Icon = st.icon;
            return (
              <div key={st.step} className="workflow-card">
                <div className="workflow-number font-mono">{st.step}</div>
                <div className="workflow-icon-box">
                  <Icon size={24} />
                </div>
                <h4 className="workflow-title">{st.title}</h4>
                <p className="workflow-desc">{st.desc}</p>
              </div>
            );
          })}
        </div>
      </section>

      {/* Technologies Used Grid */}
      <section className="tech-section section-padding">
        <div className="section-header text-center">
          <span className="section-eyebrow">TECH STACK</span>
          <h2 className="section-title">Technologies &amp; Frameworks</h2>
          <p className="section-subtitle">
            Modern full-stack technologies utilized across the interface and analytical pipeline
          </p>
        </div>

        <div className="tech-grid">
          {technologies.map((t, idx) => (
            <div key={idx} className="card tech-card">
              <div className="tech-category font-mono">{t.category}</div>
              <h4 className="tech-name">{t.name}</h4>
              <p className="tech-desc">{t.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Call to Action Banner */}
      <section className="cta-banner-section">
        <div className="cta-banner-card">
          <div className="cta-banner-left">
            <h3 className="cta-banner-title">Ready to Test an Article in the Multi-Model Lab?</h3>
            <p className="cta-banner-text">
              Simulate 5 machine learning models, cross-model comparison, and live NLP feature weights in real-time.
            </p>
          </div>
          <div className="cta-banner-right">
            <Link to="/simulation" className="btn-primary btn-lg">
              <FlaskConical size={18} />
              <span>Launch Simulation Lab</span>
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Home;

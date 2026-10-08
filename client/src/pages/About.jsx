import React from 'react';
import { 
  BookOpen, 
  BrainCircuit, 
  Binary, 
  Cpu, 
  ShieldAlert, 
  Layers, 
  CheckCircle2, 
  GitBranch,
  Terminal,
  FileCode2,
  Database
} from 'lucide-react';

export const About = () => {
  return (
    <div className="page-container about-page">
      {/* Page Header */}
      <div className="page-header">
        <div className="page-header-content">
          <div className="page-badge">
            <BookOpen size={14} />
            <span>Documentation</span>
          </div>
          <h1 className="page-title">About</h1>
          <p className="page-subtitle">
            Understanding the machine learning pipeline and classification methodology.
          </p>
        </div>
      </div>

      {/* Section 1: The Disinformation Challenge */}
      <div className="about-section card">
        <div className="card-header-styled">
          <div className="card-header-left">
            <div className="card-icon-badge text-danger">
              <ShieldAlert size={20} />
            </div>
            <div>
              <h3 className="card-title">The Challenge</h3>
              <p className="card-subtitle">Why automated news verification matters</p>
            </div>
          </div>
        </div>
        <div className="about-card-body">
          <p>
            The rapid spread of misinformation through digital media and social platforms poses significant challenges to public discourse and decision-making. Fabricated content can manipulate opinions, disrupt markets, and undermine trust in legitimate information sources.
          </p>
          <p>
            Manual fact-checking cannot keep pace with the volume of content published daily. Machine learning provides scalable, automated verification by analyzing textual patterns, writing style, and linguistic features that distinguish authentic journalism from fabricated content.
          </p>
        </div>
      </div>

      {/* Section 2: NLP Preprocessing Pipeline */}
      <div className="about-section card" id="pipeline">
        <div className="card-header-styled">
          <div className="card-header-left">
            <div className="card-icon-badge text-primary">
              <Layers size={20} />
            </div>
            <div>
              <h3 className="card-title">Text Processing Pipeline</h3>
              <p className="card-subtitle">Converting text into machine-readable features</p>
            </div>
          </div>
        </div>
        <div className="about-card-body">
          <p>
            Machine learning models require numerical inputs. Our pipeline transforms text through these stages:
          </p>
          
          <div className="pipeline-steps-grid">
            <div className="pipeline-item">
              <span className="pipeline-step-num font-mono">Step 1</span>
              <h4 className="pipeline-step-title">Tokenization</h4>
              <p className="pipeline-step-desc">Break text into individual words and standardize formatting.</p>
            </div>

            <div className="pipeline-item">
              <span className="pipeline-step-num font-mono">Step 2</span>
              <h4 className="pipeline-step-title">Noise Removal</h4>
              <p className="pipeline-step-desc">Filter common words like 'the', 'is', 'at' that don't carry meaning.</p>
            </div>

            <div className="pipeline-item">
              <span className="pipeline-step-num font-mono">Step 3</span>
              <h4 className="pipeline-step-title">Lemmatization</h4>
              <p className="pipeline-step-desc">Reduce words to their base form (e.g., 'reporting' → 'report').</p>
            </div>

            <div className="pipeline-item">
              <span className="pipeline-step-num font-mono">Step 4</span>
              <h4 className="pipeline-step-title">Feature Extraction</h4>
              <p className="pipeline-step-desc">Extract single words and two-word phrases to capture context.</p>
            </div>
          </div>
        </div>
      </div>

      {/* Section 3: TF-IDF Mathematical Formulation */}
      <div className="about-section card" id="tf-idf">
        <div className="card-header-styled">
          <div className="card-header-left">
            <div className="card-icon-badge text-amber">
              <Binary size={20} />
            </div>
            <div>
              <h3 className="card-title">TF-IDF Vectorization</h3>
              <p className="card-subtitle">Feature weighting for text analysis</p>
            </div>
          </div>
        </div>
        <div className="about-card-body">
          <p>
            TF-IDF measures word importance by balancing frequency within a document against rarity across all documents:
          </p>

          <div className="math-formula-box font-mono">
            <div className="formula-row">
              <span className="formula-label">Term Frequency (TF):</span>
              <span className="formula-code">TF(t, d) = (Count of term t in document d) / (Total words in document d)</span>
            </div>
            <div className="formula-row">
              <span className="formula-label">Inverse Document Frequency (IDF):</span>
              <span className="formula-code">IDF(t, D) = log( (1 + Total documents N) / (1 + Documents containing term t) ) + 1</span>
            </div>
            <div className="formula-row formula-highlight">
              <span className="formula-label">Composite TF-IDF Weight:</span>
              <span className="formula-code">TF-IDF(t, d, D) = TF(t, d) &times; IDF(t, D)</span>
            </div>
          </div>

          <p className="formula-explanation">
            <strong>Key Intuition:</strong> Words that occur frequently in a specific article (high TF) but rarely across the general corpus (high IDF) receive large weights, capturing distinguishing vocabulary signatures indicative of deceptive reporting.
          </p>
        </div>
      </div>

      {/* Section 4: Machine Learning Classification Models */}
      <div className="about-section card" id="models">
        <div className="card-header-styled">
          <div className="card-header-left">
            <div className="card-icon-badge text-success">
              <Cpu size={20} />
            </div>
            <div>
              <h3 className="card-title">Supervised Classification Algorithms Evaluated</h3>
              <p className="card-subtitle">Empirical performance comparison of standard classifiers</p>
            </div>
          </div>
        </div>
        <div className="about-card-body">
          <div className="models-detail-grid">
            {/* Model 1 */}
            <div className="model-detail-card">
              <div className="model-card-top">
                <span className="model-name-large">Logistic Regression</span>
                <span className="model-accuracy-badge font-mono">C=1.0</span>
              </div>
              <p className="model-detail-desc">
                A probabilistic linear classifier utilizing the standard Sigmoid activation &sigma;(z) = 1 / (1 + e<sup>-z</sup>) to yield calibrated confidence probabilities between 0 and 1.
              </p>
            </div>

            {/* Model 2 */}
            <div className="model-detail-card">
              <div className="model-card-top">
                <span className="model-name-large">Multinomial Naive Bayes</span>
                <span className="model-accuracy-badge font-mono">&alpha;=1.0</span>
              </div>
              <p className="model-detail-desc">
                Applies Bayes' theorem under the assumption of conditional word feature independence, providing lightning-fast computation and strong baseline metrics.
              </p>
            </div>

            {/* Model 3 */}
            <div className="model-detail-card">
              <div className="model-card-top">
                <span className="model-name-large">Linear Support Vector Machine</span>
                <span className="model-accuracy-badge font-mono">C=1.0, Linear</span>
              </div>
              <p className="model-detail-desc">
                Optimizes a maximum-margin hyperplane in high-dimensional sparse TF-IDF vector space, effectively separating truthful from deceptive lexical clusters.
              </p>
            </div>

            {/* Model 4 */}
            <div className="model-detail-card">
              <div className="model-card-top">
                <span className="model-name-large">Decision Tree</span>
                <span className="model-accuracy-badge font-mono">Gini Impurity</span>
              </div>
              <p className="model-detail-desc">
                Non-parametric rule-based partitioning classifier that splits the TF-IDF feature space based on information gain and Gini impurity metrics.
              </p>
            </div>

            {/* Model 5 */}
            <div className="model-detail-card">
              <div className="model-card-top">
                <span className="model-name-large">Random Forest</span>
                <span className="model-accuracy-badge font-mono">100 Trees</span>
              </div>
              <p className="model-detail-desc">
                Ensemble meta-estimator that fits a multitude of randomized decision trees on sub-samples and uses majority voting to control overfitting and deliver peak accuracy.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Section 5: Full-Stack MERN + ML Architecture */}
      <div className="about-section card">
        <div className="card-header-styled">
          <div className="card-header-left">
            <div className="card-icon-badge text-primary">
              <GitBranch size={20} />
            </div>
            <div>
              <h3 className="card-title">System Architecture &amp; Backend Communication</h3>
              <p className="card-subtitle">Seamless integration blueprint between React UI and Python ML services</p>
            </div>
          </div>
        </div>
        <div className="about-card-body">
          <div className="arch-flow-diagram">
            <div className="arch-node">
              <FileCode2 size={24} className="text-primary" />
              <div className="arch-node-title">React 18 + Vite UI</div>
              <div className="arch-node-sub font-mono">Port 5173 / Axios</div>
            </div>
            <div className="arch-arrow">&rarr; POST /api/predictions &rarr;</div>
            <div className="arch-node">
              <Terminal size={24} className="text-success" />
              <div className="arch-node-title">Express.js REST API</div>
              <div className="arch-node-sub font-mono">Port 5000 / Controller</div>
            </div>
            <div className="arch-arrow">&rarr; ChildProcess / FastAPI &rarr;</div>
            <div className="arch-node">
              <Cpu size={24} className="text-amber" />
              <div className="arch-node-title">Python ML Engine</div>
              <div className="arch-node-sub font-mono">Scikit-Learn / Pickle</div>
            </div>
          </div>

          <p className="arch-explanation">
            The frontend communicates through a centralized <code>src/services/api.js</code> service module configured with environment variable <code>VITE_API_URL</code>. When the user analyzes an article, the UI dispatches a JSON payload containing <code>{`{ title, text }`}</code> and receives standardized verdict objects with confidence ratings, model metadata, and linguistic analysis.
          </p>
        </div>
      </div>
    </div>
  );
};

export default About;

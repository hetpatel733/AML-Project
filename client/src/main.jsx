import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css';
import { DatasetProvider } from './context/DatasetContext';

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <DatasetProvider>
      <App />
    </DatasetProvider>
  </React.StrictMode>
);

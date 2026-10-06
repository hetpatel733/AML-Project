import React, { createContext, useState, useContext, useEffect } from 'react';

const DatasetContext = createContext();

export const DatasetProvider = ({ children }) => {
  const [selectedDataset, setSelectedDataset] = useState(() => {
    return localStorage.getItem('selectedDataset') || 'isot';
  });

  useEffect(() => {
    localStorage.setItem('selectedDataset', selectedDataset);
  }, [selectedDataset]);

  return (
    <DatasetContext.Provider value={{ selectedDataset, setSelectedDataset }}>
      {children}
    </DatasetContext.Provider>
  );
};

export const useDataset = () => useContext(DatasetContext);

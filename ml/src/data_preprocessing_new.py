# =============================================================================
# LIAR Dataset Label Mappings
# =============================================================================

# LIAR 6-class to binary mapping (for binary classification)
LIAR_TO_BINARY = {
    'true': 'real',
    'mostly-true': 'real', 
    'half-true': 'real',
    'barely-true': 'fake',
    'false': 'fake',
    'pants-fire': 'fake'
}

# LIAR label to numeric (for multi-class)
LIAR_LABEL_NUMERIC = {
    'true': 0,
    'mostly-true': 1,
    'half-true': 2,
    'barely-true': 3,
    'false': 4,
    'pants-fire': 5
}

# Binary label to numeric
BINARY_LABEL_MAP = {'real': 0, 'fake': 1}


def load_isot_dataset(test_size: float = 0.2, random_state: int = 42):
    """Load and split ISOT dataset (Fake/True binary classification)."""
    from src.config import ISOT_FAKE_CSV, ISOT_TRUE_CSV
    
    fake_df = pd.read_csv(ISOT_FAKE_CSV)
    true_df = pd.read_csv(ISOT_TRUE_CSV)
    
    fake_df['label'] = 1  # FAKE
    true_df['label'] = 0  # REAL/TRUE
    
    df = pd.concat([fake_df, true_df], ignore_index=True)
    df = df.dropna(subset=['text', 'title'])
    df['text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    
    X = df['text'].values
    y = df['label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test


def load_liar_dataset(test_size: float = 0.2, random_state: int = 42, binary: bool = True):
    """Load and split LIAR dataset.
    
    Args:
        binary: If True, map to binary (fake/real). If False, keep 6-class.
    """
    from src.config import LIAR_TRAIN_TSV, LIAR_VALID_TSV, LIAR_TEST_TSV
    
    columns = ['id', 'label', 'statement', 'subject', 'speaker', 'job', 'state', 
               'party', 'barely_true', 'false', 'half_true', 'mostly_true', 
               'pants_fire', 'context']
    
    train_df = pd.read_csv(LIAR_TRAIN_TSV, sep='\t', names=columns, header=0)
    valid_df = pd.read_csv(LIAR_VALID_TSV, sep='\t', names=columns, header=0)
    test_df = pd.read_csv(LIAR_TEST_TSV, sep='\t', names=columns, header=0)
    
    df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    df = df.dropna(subset=['statement'])
    
    if binary:
        df['label'] = df['label'].map(LIAR_TO_BINARY)
    else:
        df['label'] = df['label'].map(LIAR_LABEL_NUMERIC)
    
    df = df.dropna(subset=['label'])
    df['text'] = df['statement'].fillna('')
    
    X = df['text'].values
    y = df['label'].values
    
    stratify_param = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
    )
    
    return X_train, X_test, y_train, y_test


def load_dataset(dataset: str, test_size: float = 0.2, random_state: int = 42):
    """Unified dataset loader.
    
    Args:
        dataset: 'isot' or 'liar'
        test_size: Test split ratio
        random_state: Random seed
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    dataset = dataset.lower()
    if dataset == 'isot':
        return load_isot_dataset(test_size, random_state)
    elif dataset == 'liar':
        return load_liar_dataset(test_size, random_state, binary=True)
    else:
        raise ValueError(f'Unknown dataset: {dataset}. Use "isot" or "liar"')
class Matrix:

    @staticmethod
    def save(m_path: str):        
        return np.save(m_path)

    @staticmethod
    def load(m_path: str):
        return np.load(rm_path)
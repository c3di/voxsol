class Material {

public:
    Material(unsigned int id, double rho, double eModul, double poissonRatio, double yieldStrength);
    ~Material();

    const unsigned int m_id;
    const double m_lambda;
    const double m_mu;

private:

    double calculateMu(double eModul, double poissonRatio);
    double calculateLambda(double eModul, double poissonRatio);

};

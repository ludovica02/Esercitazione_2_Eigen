#include <iostream>
#include "Eigen/Eigen"

using namespace std;
using namespace Eigen;

VectorXd SolveSystemPALU(const MatrixXd& A,
                     const VectorXd& b,
                     const VectorXd& exactSolution,
                     double& errRelPALU)
{
    VectorXd solutionPALU = A.fullPivLu().solve(b);
    errRelPALU = (solutionPALU - exactSolution).norm() / exactSolution.norm();

    return solutionPALU;
}

VectorXd SolveSystemQR(const MatrixXd& A,
                   const VectorXd& b,
                   const VectorXd& exactSolution,
                   double& errRelQR)
{
    VectorXd solutionQR = A.householderQr().solve(b);
    errRelQR = (solutionQR - exactSolution).norm() / exactSolution.norm();

    return solutionQR;
}

int main()
{

    Vector2d exactSolution = {-1.0e+0, -1.0e+00};

    Matrix2d A1;
    A1 << 5.547001962252291e-01, -3.770900990025203e-02,
        8.320502943378437e-01, -9.992887623566787e-01;

    Vector2d b1 = {-5.169911863249772e-01, 1.672384680188350e-01};

    Vector2d solutionPALU1, solutionQR1;

    double errRelPALU1, errRelQR1;

    solutionPALU1 = SolveSystemPALU(A1, b1, exactSolution, errRelPALU1);
    solutionQR1 = SolveSystemQR(A1, b1, exactSolution, errRelQR1);

    cout << scientific << "System 1: Relative error PALU: " << errRelPALU1 << " , " << "Relative Error QR: " << errRelQR1 << endl;

    Matrix2d A2;
    A2 << 5.547001962252291e-01, -5.540607316466765e-01,
        8.320502943378437e-01, -8.324762492991313e-01;

    Vector2d b2 = {-6.394645785530173e-04, 4.259549612877223e-04};

    Vector2d solutionPALU2, solutionQR2;

    double errRelPALU2, errRelQR2;

    solutionPALU2 = SolveSystemPALU(A2, b2, exactSolution, errRelPALU2);
    solutionQR2 = SolveSystemQR(A2, b2, exactSolution, errRelQR2);

    cout << scientific << "System 2: Relative error PALU: " << errRelPALU2 << " , " << "Relative Error QR: " << errRelQR2 << endl;

    Matrix2d A3;
    A3 << 5.547001962252291e-01, -5.547001955851905e-01,
        8.320502943378437e-01, -8.320502947645361e-01;

    Vector2d b3 = {-6.400391328043042e-10, 4.266924591433963e-10};

    Vector2d solutionPALU3, solutionQR3;

    double errRelPALU3, errRelQR3;

    solutionPALU3 = SolveSystemPALU(A3, b3, exactSolution, errRelPALU3);
    solutionQR3 = SolveSystemQR(A3, b3, exactSolution, errRelQR3);

    cout << scientific << "System 3: Relative error PALU: " << errRelPALU3 << " , " << "Relative Error QR: " << errRelQR3 << endl;

    return 0;
}

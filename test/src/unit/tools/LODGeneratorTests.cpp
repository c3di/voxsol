#include "stdafx.h"
#include "gtest/gtest.h"
#include "gtest/internal/gtest-internal.h"
#include "libmmv/math/Vec3.h"
#include "helpers/Templates.h"
#include "problem/boundaryconditions/NeumannBoundary.h"
#include "tools/LODGenerator.h"

class LODGeneratorTests : public ::testing::Test {

public:
    LODGeneratorTests() {}
    ~LODGeneratorTests() {}

    bool closeEqual(const libmmv::Vec3<REAL>& a, const libmmv::Vec3<REAL>& b) {
        for (unsigned int i = 0; i < 3; i++) {
            const testing::internal::FloatingPoint<REAL> lhs(a[i]), rhs(b[i]);
            if (!lhs.AlmostEquals(rhs)) {
                return false;
            }
        }
        return true;
    }

    void SetUp() override
    {

    }

    void TearDown() override
    {

    }

};

TEST_F(LODGeneratorTests, DirichletFromFineToCoarse) {
    DiscreteProblem fineProblem = Templates::Problem::STEEL(libmmv::Vec3ui(4, 4, 4));
    DiscreteProblem coarseProblem = Templates::Problem::STEEL(libmmv::Vec3ui(2, 2, 2));

    DirichletBoundary fixed(DirichletBoundary::FIXED_ALL);
    DirichletBoundary fixedX(DirichletBoundary::FIXED_X);
    DirichletBoundary fixedY(DirichletBoundary::FIXED_Y);

    fineProblem.setDirichletBoundaryAtVertex(VertexCoordinate(4, 2, 0), fixed);
    fineProblem.setDirichletBoundaryAtVertex(VertexCoordinate(1, 1, 0), fixedX);
    fineProblem.setDirichletBoundaryAtVertex(VertexCoordinate(2, 1, 0), fixedY);

    LODGenerator lodgen;
    lodgen.populateCoarserLevelProblem(&coarseProblem, &fineProblem);

    std::unordered_map<unsigned int, DirichletBoundary>* coarseDirichletMap = coarseProblem.getDirichletBoundaryMap();

    ASSERT_EQ(coarseDirichletMap->size(), 5) << "Expected fine dirichlet boundaries to be extrapolated to neighbors in coarse problem";

    DirichletBoundary coarse210 = coarseProblem.getDirichletBoundaryAtVertex(VertexCoordinate(2, 1, 0));
    ASSERT_EQ(coarse210.fixed, DirichletBoundary::FIXED_ALL) << "Expected dirichlet boundary to be projected directly onto the same vertex in the coarse problem";

    DirichletBoundary coarse100 = coarseProblem.getDirichletBoundaryAtVertex(VertexCoordinate(1, 0, 0));
    ASSERT_TRUE(coarse100.isXFixed() && coarse100.isYFixed() && !coarse100.isZFixed()) << "Expected fixed X and Y boundaries to be combined";

    DirichletBoundary coarse000 = coarseProblem.getDirichletBoundaryAtVertex(VertexCoordinate(0, 0, 0));
    ASSERT_EQ(coarse000.fixed, DirichletBoundary::FIXED_X) << "Expected fixed X boundary to be projected to surviving neighboring vertices in coarse solution";
}

TEST_F(LODGeneratorTests, NeumannFromFineToCoarse) {
    DiscreteProblem fineProblem = Templates::Problem::STEEL(libmmv::Vec3ui(4, 4, 4));
    DiscreteProblem coarseProblem = Templates::Problem::STEEL(libmmv::Vec3ui(2, 2, 2));

    NeumannBoundary neumann5x(libmmv::Vec3<REAL>(5, 0, 0));
    NeumannBoundary neumann5y(libmmv::Vec3<REAL>(0, 5, 0));
    NeumannBoundary neumann5z(libmmv::Vec3<REAL>(0, 0, 5));
    NeumannBoundary neumann10all(libmmv::Vec3<REAL>(10, 10, 10));

    fineProblem.setNeumannBoundaryAtVertex(VertexCoordinate(1, 1, 0), neumann5x);
    fineProblem.setNeumannBoundaryAtVertex(VertexCoordinate(2, 1, 0), neumann5y);
    fineProblem.setNeumannBoundaryAtVertex(VertexCoordinate(4, 0, 0), neumann10all);
    fineProblem.setNeumannBoundaryAtVertex(VertexCoordinate(4, 1, 0), neumann5z);

    LODGenerator lodgen;
    lodgen.populateCoarserLevelProblem(&coarseProblem, &fineProblem);

    std::unordered_map<unsigned int, NeumannBoundary>* coarseNeumannMap = coarseProblem.getNeumannBoundaryMap();

    ASSERT_EQ(coarseNeumannMap->size(), 6) << "Expected fine neumann boundaries to be extrapolated to neighbors in coarse problem";

    // fine coord 1,1,0 should map to coarse coords 000, 100, 010, 110 which should each get 1/4th the stress
    NeumannBoundary coarse000 = coarseProblem.getNeumannBoundaryAtVertex(VertexCoordinate(0, 0, 0));
    ASSERT_TRUE(closeEqual(coarse000.stress, libmmv::Vec3<REAL>(asREAL(5) / asREAL(4), 0, 0)));

    // fine coord 2,1,0 should map to coarse coords 100, 110 which should each get 1/2 the Y stress, plus the 1/4 X stress from fine coord 1,1,0
    NeumannBoundary coarse100 = coarseProblem.getNeumannBoundaryAtVertex(VertexCoordinate(1, 0, 0));
    ASSERT_TRUE(closeEqual(coarse100.stress, libmmv::Vec3<REAL>(asREAL(5) / asREAL(4), asREAL(2.5), 0)));

    // fine coord 4,0,0 should map to coarse coord 200 which should get the full 10 stress in XYZ plus the 1/2 Z stress from fine coord 4,1,0
    NeumannBoundary coarse200 = coarseProblem.getNeumannBoundaryAtVertex(VertexCoordinate(2, 0, 0));
    ASSERT_TRUE(closeEqual(coarse200.stress, libmmv::Vec3<REAL>(asREAL(10), asREAL(10), asREAL(10) + asREAL(2.5))));
}

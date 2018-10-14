
#include <list>
#include <memory>


#include "BuildIrTests.h"
#include "ChrisSelection.h"
#include "IrBuilder.h"
#include "MaterializationModeNamedSet.h"
#include "MaterializationModeAlgo.h"
#include "SharedEmployee.h"
#include "QueryNodeIr.h"
#include "Selection.h"
#include "SelectionIr.h"
#include "Set.h"
#include "SetExpressionIrAlgo.h"
#include "StringSelection.h"
#include "SourceSetNameIr.h"
#include "Supervisor.h"
#include "ProjectionIr.h"
#include "QueryGraphIr.h"
#include "QueryOutput.h"

using std::dynamic_pointer_cast;
using std::list;
using std::function;
using std::string;

using pdb::Handle;
using pdb::SimpleLambda;
using pdb::makeObject;
using pdb::Object;
using pdb::QueryOutput;
using pdb::Selection;
using pdb::Supervisor;
using pdb::Set;
using pdb::unsafeCast;

using pdb_detail::buildIr;
using pdb_detail::buildIrSingle;
using pdb_detail::MaterializationModeAlgo;
using pdb_detail::MaterializationModeNone;
using pdb_detail::MaterializationModeNamedSet;
using pdb_detail::QueryGraphIrPtr;
using pdb_detail::ProjectionIr;
using pdb_detail::RecordPredicateIr;
// using pdb_detail::RecordProjectionIr;
using pdb_detail::SelectionIr;
using pdb_detail::SetExpressionIr;
using pdb_detail::SetExpressionIrAlgo;
using pdb_detail::SourceSetNameIr;


namespace pdb_tests {
bool isProjection(shared_ptr<SetExpressionIr> expression) {
    class IsProjection : public SetExpressionIrAlgo {
    public:
        void forProjection(ProjectionIr& recordProjection) {
            correct = true;
        }

        void forSelection(SelectionIr& selection) {}

        void forSourceSetName(SourceSetNameIr& setName) {}

        bool correct = false;

    } isProjection;

    expression->execute(isProjection);

    return isProjection.correct;
}

bool isSelection(shared_ptr<SetExpressionIr> expression) {
    class IsSelection : public SetExpressionIrAlgo {
    public:
        void forProjection(ProjectionIr& recordProjection) {}

        void forSelection(SelectionIr& selection) {
            correct = true;
        }

        void forSourceSetName(SourceSetNameIr& setName) {}

        bool correct = false;

    } isSelection;

    expression->execute(isSelection);

    return isSelection.correct;
}

bool isSourceSetName(shared_ptr<SetExpressionIr> expression) {
    class IsSourceSetName : public SetExpressionIrAlgo {
    public:
        void forProjection(ProjectionIr& recordProjection) {}

        void forSelection(SelectionIr& selection) {}

        void forSourceSetName(SourceSetNameIr& setName) {
            correct = true;
        }

        bool correct = false;
    } isSourceSetName;

    expression->execute(isSourceSetName);

    return isSourceSetName.correct;
}

class NamedSetExtractor : public MaterializationModeAlgo {
public:
    void forNone(MaterializationModeNone& mode) {}

    void forNamedSet(MaterializationModeNamedSet& mode) {
        outDatabaseName = mode.getDatabaseName();
        outSetName = mode.getSetName();
    }

    string outDatabaseName = "";

    string outSetName = "";
};

/**
 * Create a simple pdb::Selection over a set name and ensure the built QueryIr object is correct.
 */
void testBuildIrSelection1(UnitTest& qunit) {

    /**
     * Create the selection to translate.
     */
    class MySelectionType : public Selection<Object, Object> {

    public:
        virtual SimpleLambda<bool> getSelection(Handle<Object>& in) override {
            return SimpleLambda<bool>([]() { return true; });
        }

        virtual SimpleLambda<Handle<Object>> getProjection(Handle<Object>& in) override {
            return SimpleLambda<Handle<Object>>([]() { return nullptr; });
        }

        // over-ridden by the user so they can supply the selection on projection
        // temporarily added by Jia: for testing pipeline execution for logical plan with
        // pushing-down projection
        SimpleLambda<bool> getProjectionSelection(Handle<Object>& in) override {}


        ENABLE_DEEP_COPY
    };

    Handle<MySelectionType> selection = makeObject<MySelectionType>();
    {
        Handle<Set<Object>> selectionInput =
            makeObject<Set<Object>>("inputDatabaseName", "inputSetName");
        selection->setInput(selectionInput);
    }

    selection->setDBName("outputDatabaseName");
    selection->setSetName("outputSetName");


    /**
     * Translate MySelection to QueryNodeIr.
     *
     */
    QueryGraphIrPtr queryGraph = buildIrSingle(selection);


    /**
     * Test that projection is the only sink node.
     */
    QUNIT_IS_EQUAL(1, queryGraph->getSinkNodeCount());
    shared_ptr<SetExpressionIr> querySink = queryGraph->getSinkNode(0);

    QUNIT_IS_FALSE(querySink->getMaterializationMode()->isNone());

    NamedSetExtractor outSetExtractor;

    querySink->getMaterializationMode()->execute(outSetExtractor);

    QUNIT_IS_EQUAL("outputDatabaseName", outSetExtractor.outDatabaseName);
    QUNIT_IS_EQUAL("outputSetName", outSetExtractor.outSetName);


    QUNIT_IS_TRUE(isProjection(querySink));

    shared_ptr<ProjectionIr> projectionIr = dynamic_pointer_cast<ProjectionIr>(querySink);

    /**
     * Test that the input to the projection is the selection
     */
    shared_ptr<SetExpressionIr> projectionInput = projectionIr->getInputSet();

    QUNIT_IS_TRUE(isSelection(projectionInput));

    shared_ptr<SelectionIr> selectionIr = dynamic_pointer_cast<SelectionIr>(projectionInput);

    /**
     * Test that the input to the selection is the set "setname" in the database "databasename"
     */
    shared_ptr<SetExpressionIr> selectionInput = selectionIr->getInputSet();

    QUNIT_IS_TRUE(isSourceSetName(selectionInput));

    shared_ptr<SourceSetNameIr> selectionSetName =
        dynamic_pointer_cast<SourceSetNameIr>(selectionInput);

    QUNIT_IS_EQUAL("inputDatabaseName", selectionSetName->getDatabaseName());
    QUNIT_IS_EQUAL("inputSetName", selectionSetName->getSetName());
}

void testBuildIrSelection2(UnitTest& qunit) {
    /**
     * A user defined type to prove we can work over types other than object.
     */
    class Zebra : public Object {
        ENABLE_DEEP_COPY
    };


    class MySelectionType : public Selection<Zebra, Zebra> {

    public:
        SimpleLambda<bool> getSelection(Handle<Zebra>& in) override {
            return SimpleLambda<bool>([]() { return true; });
        }

        SimpleLambda<Handle<Zebra>> getProjection(Handle<Zebra>& in) override {
            return SimpleLambda<Handle<Zebra>>([]() { return nullptr; });
        }

        // over-ridden by the user so they can supply the selection on projection
        // temporarily added by Jia: for testing pipeline execution for logical plan with
        // pushing-down projection
        SimpleLambda<bool> getProjectionSelection(Handle<Zebra>& in) override {}


        ENABLE_DEEP_COPY
    };


    /**
     * Setup a user query graph that looks like:
     *
     * (outputSet1)<--(selection2)<--(selection1)<--(inputSet)
     *                               |
     *                (outputSet2)<---
     *
     */
    Handle<Set<Zebra>> sourceSet = makeObject<Set<Zebra>>("somedb", "inputSetName");

    Handle<MySelectionType> selection1 = makeObject<MySelectionType>();
    selection1->setInput(sourceSet);

    Handle<MySelectionType> selection2 = makeObject<MySelectionType>();
    selection2->setInput(selection1);

    Handle<QueryOutput<Zebra>> outputSet1 =
        makeObject<QueryOutput<Zebra>>("somedb", "outputSetName1", selection1);


    Handle<QueryOutput<Zebra>> outputSet2 =
        makeObject<QueryOutput<Zebra>>("somedb", "outputSetName2", selection2);


    /**
     * Test translation of the user query graph with sinks outputSet1 and outputSet2 to a logical
     * graph.
     *
     * The logical graph should have the form:
     *
     * (querySink) <- (internalNode1) <- (internalNode2) <- (internalNode3) <- (sourceNode)
     *
     *
     * With some following details:
     *
     * querySink: type = ProjectionIr, materialization? = yes (outputSetName2)
     *
     * internalNode1: type = SelectionIr, materialization? = no
     *
     * internalNode2: type = ProjectionIr, materialization? = yes (outputSetName1)
     *
     * internalNode3: type = SelectionIr, materialization? = no
     *
     * sourceNode: type = SourceSetNameIr, materialization? = no
     */
    Handle<Vector<Handle<QueryBase>>> sinks = makeObject<Vector<Handle<QueryBase>>>();
    sinks->push_back(outputSet2);
    sinks->push_back(outputSet1);

    QueryGraphIrPtr queryGraph = buildIr(sinks);

    QUNIT_IS_EQUAL(1, queryGraph->getSinkNodeCount());
    shared_ptr<SetExpressionIr> sinkNode0 = queryGraph->getSinkNode(0);

    // querySink checks
    QUNIT_IS_FALSE(sinkNode0->getMaterializationMode()->isNone());

    NamedSetExtractor outSetExtractor;
    sinkNode0->getMaterializationMode()->execute(outSetExtractor);
    QUNIT_IS_EQUAL("somedb", outSetExtractor.outDatabaseName);
    QUNIT_IS_EQUAL("outputSetName2", outSetExtractor.outSetName);

    QUNIT_IS_TRUE(isProjection(sinkNode0));

    shared_ptr<ProjectionIr> sinkNode0Typed = dynamic_pointer_cast<ProjectionIr>(sinkNode0);

    // internalNode1 checks
    shared_ptr<SetExpressionIr> internalNode1 = sinkNode0Typed->getInputSet();

    QUNIT_IS_TRUE(internalNode1->getMaterializationMode()->isNone());
    QUNIT_IS_TRUE(isSelection(internalNode1));

    shared_ptr<SelectionIr> internalNode1Typed = dynamic_pointer_cast<SelectionIr>(internalNode1);

    // internalNode2 checks
    shared_ptr<SetExpressionIr> internalNode2 = internalNode1Typed->getInputSet();

    QUNIT_IS_FALSE(internalNode2->getMaterializationMode()->isNone());

    internalNode2->getMaterializationMode()->execute(outSetExtractor);
    QUNIT_IS_EQUAL("somedb", outSetExtractor.outDatabaseName);
    QUNIT_IS_EQUAL("outputSetName1", outSetExtractor.outSetName);

    QUNIT_IS_TRUE(isProjection(internalNode2));

    shared_ptr<ProjectionIr> internalNode2Typed = dynamic_pointer_cast<ProjectionIr>(internalNode2);

    // internalNode3 checks
    shared_ptr<SetExpressionIr> internalNode3 = internalNode2Typed->getInputSet();

    QUNIT_IS_TRUE(internalNode3->getMaterializationMode()->isNone());
    QUNIT_IS_TRUE(isSelection(internalNode3));

    shared_ptr<SelectionIr> internalNode3Typed = dynamic_pointer_cast<SelectionIr>(internalNode3);

    // sourceNode checks
    shared_ptr<SetExpressionIr> sourceNode = internalNode3Typed->getInputSet();

    QUNIT_IS_TRUE(isSourceSetName(sourceNode));
    QUNIT_IS_TRUE(sourceNode->getMaterializationMode()->isNone());

    shared_ptr<SourceSetNameIr> sourceNodeTyped = dynamic_pointer_cast<SourceSetNameIr>(sourceNode);

    QUNIT_IS_EQUAL("somedb", sourceNodeTyped->getDatabaseName());
    QUNIT_IS_EQUAL("inputSetName", sourceNodeTyped->getSetName());
}

void testBuildIrSelection3(UnitTest& qunit) {

    /**
     * Setup a user query graph that looks like:
     *
     * (outputSet2)<--(selection2)<--(selection1)<--(inputSet)
     *                               |
     *                (outputSet1)<---
     *
     */
    Handle<Set<SharedEmployee>> myInputSet =
        makeObject<Set<SharedEmployee>>("chris_db", "chris_set");
    Handle<ChrisSelection> myFirstSelect = makeObject<ChrisSelection>();
    myFirstSelect->setInput(myInputSet);
    Handle<StringSelection> mySecondSelect = makeObject<StringSelection>();
    mySecondSelect->setInput(myFirstSelect);
    Handle<QueryOutput<String>> outputOne =
        makeObject<QueryOutput<String>>("chris_db", "output_set1", myFirstSelect);
    Handle<QueryOutput<String>> outputTwo =
        makeObject<QueryOutput<String>>("chris_db", "output_set2", mySecondSelect);
    Handle<Vector<Handle<QueryBase>>> queries = makeObject<Vector<Handle<QueryBase>>>();
    queries->push_back(outputOne);
    queries->push_back(outputTwo);


    /**
     * Test translation of the user query graph with sinks outputSet1 and outputSet2 to a logical
     * graph.
     *
     * The logical graph should have the form:
     *
     * (querySink) <- (internalNode1) <- (internalNode2) <- (internalNode3) <- (sourceNode)
     *
     *
     * With some following details:
     *
     * querySink: type = ProjectionIr, materialization? = yes (outputSetName2)
     *
     * internalNode1: type = SelectionIr, materialization? = no
     *
     * internalNode2: type = ProjectionIr, materialization? = yes (outputSetName1)
     *
     * internalNode3: type = SelectionIr, materialization? = no
     *
     * sourceNode: type = SourceSetNameIr, materialization? = no
     */
    Handle<Vector<Handle<QueryBase>>> sinks = makeObject<Vector<Handle<QueryBase>>>();
    sinks->push_back(outputTwo);
    sinks->push_back(outputOne);

    QueryGraphIrPtr queryGraph = buildIr(sinks);

    QUNIT_IS_EQUAL(1, queryGraph->getSinkNodeCount());
    shared_ptr<SetExpressionIr> sinkNode0 = queryGraph->getSinkNode(0);

    // querySink checks
    QUNIT_IS_FALSE(sinkNode0->getMaterializationMode()->isNone());

    NamedSetExtractor outSetExtractor;
    sinkNode0->getMaterializationMode()->execute(outSetExtractor);
    QUNIT_IS_EQUAL("chris_db", outSetExtractor.outDatabaseName);
    QUNIT_IS_EQUAL("output_set2", outSetExtractor.outSetName);

    QUNIT_IS_TRUE(isProjection(sinkNode0));

    shared_ptr<ProjectionIr> sinkNode0Typed = dynamic_pointer_cast<ProjectionIr>(sinkNode0);

    // internalNode1 checks
    shared_ptr<SetExpressionIr> internalNode1 = sinkNode0Typed->getInputSet();

    QUNIT_IS_TRUE(internalNode1->getMaterializationMode()->isNone());
    QUNIT_IS_TRUE(isSelection(internalNode1));

    shared_ptr<SelectionIr> internalNode1Typed = dynamic_pointer_cast<SelectionIr>(internalNode1);

    // internalNode2 checks
    shared_ptr<SetExpressionIr> internalNode2 = internalNode1Typed->getInputSet();

    QUNIT_IS_FALSE(internalNode2->getMaterializationMode()->isNone());

    internalNode2->getMaterializationMode()->execute(outSetExtractor);
    QUNIT_IS_EQUAL("chris_db", outSetExtractor.outDatabaseName);
    QUNIT_IS_EQUAL("output_set1", outSetExtractor.outSetName);

    QUNIT_IS_TRUE(isProjection(internalNode2));

    shared_ptr<ProjectionIr> internalNode2Typed = dynamic_pointer_cast<ProjectionIr>(internalNode2);

    // internalNode3 checks
    shared_ptr<SetExpressionIr> internalNode3 = internalNode2Typed->getInputSet();

    QUNIT_IS_TRUE(internalNode3->getMaterializationMode()->isNone());
    QUNIT_IS_TRUE(isSelection(internalNode3));

    shared_ptr<SelectionIr> internalNode3Typed = dynamic_pointer_cast<SelectionIr>(internalNode3);

    // sourceNode checks
    shared_ptr<SetExpressionIr> sourceNode = internalNode3Typed->getInputSet();

    QUNIT_IS_TRUE(isSourceSetName(sourceNode));
    QUNIT_IS_TRUE(sourceNode->getMaterializationMode()->isNone());

    shared_ptr<SourceSetNameIr> sourceNodeTyped = dynamic_pointer_cast<SourceSetNameIr>(sourceNode);

    QUNIT_IS_EQUAL("chris_db", sourceNodeTyped->getDatabaseName());
    QUNIT_IS_EQUAL("chris_set", sourceNodeTyped->getSetName());
}
}

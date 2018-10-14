#ifndef CHECK_EMPLOYEES_H
#define CHECK_EMPLOYEES_H

#include "Selection.h"
#include "Employee.h"
#include "Supervisor.h"
#include "PDBVector.h"
#include "PDBString.h"

// PRELOAD %CheckEmployee%

namespace pdb {

// this silly little class accepts a Supervisor, and rejects that supervisor
// if he doesn't manage anyone other than "nameToExclude".  If he does manage
// someone other than "nameToExclude", then the query will return all of the
// employees for the supervisor that don't have that name
class CheckEmployee : public Selection<Vector<Handle<Employee>>, Supervisor> {

private:
    String nameToExclude;

public:
    ENABLE_DEEP_COPY

    CheckEmployee() {}

    CheckEmployee(std::string& nameToExcludeIn) {
        nameToExclude = nameToExcludeIn;
    }

    SimpleLambda<bool> getSelection(Handle<Supervisor>& checkMe) override {
        return makeSimpleLambda(checkMe, [&]() -> bool {
            int numEmployees = checkMe->getNumEmployees();
            for (unsigned int i = 0; i < numEmployees; i++) {
                if (*(checkMe->getEmp(i)->getName()) != nameToExclude) {
                    return true;
                }
            }
            return false;
        });
    }

    SimpleLambda<bool> getProjectionSelection(Handle<Vector<Handle<Employee>>>& checkMe) override {
        return makeSimpleLambda(checkMe, [&]() -> bool {
            size_t numEmployees = checkMe->size();
            // std :: cout << "numEmployees=" << numEmployees << std :: endl;
            for (size_t i = 0; i < numEmployees; i++) {
                // std :: cout << i << ":"  << *(*checkMe)[i]->getName () << std :: endl;
                if ((*(*checkMe)[i]->getName()) != nameToExclude) {
                    return true;
                }
            }
            return false;
        });
    }

    SimpleLambda<Handle<Vector<Handle<Employee>>>> getProjection(
        Handle<Supervisor>& checkMe) override {
        return makeSimpleLambda(checkMe, [&] {
            Handle<Vector<Handle<Employee>>> returnVal = makeObject<Vector<Handle<Employee>>>(10);
            size_t numEmployees = checkMe->getNumEmployees();
            // std :: cout << "numEmployees=" << numEmployees << std :: endl;
            for (size_t i = 0; i < numEmployees; i++) {
                // std :: cout << i << ":"  << *(checkMe->getEmp (i)->getName ()) << std :: endl;
                if (*(checkMe->getEmp(i)->getName()) != nameToExclude) {
                    returnVal->push_back(checkMe->getEmp(i));
                }
            }

            return returnVal;
        });
    }
};
}
#endif

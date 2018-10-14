/*
 * CatalogStandardNodeMetadata.cc
 *
 *  Created on: Dec 8, 2016
 *      Author: carlos
 */

#include "CatalogStandardNodeMetadata.h"

using namespace std;

CatalogStandardNodeMetadata::CatalogStandardNodeMetadata() {}

CatalogStandardNodeMetadata::CatalogStandardNodeMetadata(
    string nodeIdValue, string nodeIPValue, int nodePortValue,
    string nodeNameValue, string nodeTypeValue, int nodeStatusValue)
    : nodeId(nodeIdValue), nodeIP(nodeIPValue), nodePort(nodePortValue),
      nodeName(nodeNameValue), nodeType(nodeTypeValue),
      nodeStatus(nodeStatusValue) {
  nodeAddress =
      std::string(nodeIP) + std::string(":") + std::to_string(nodePort);
}

// Copy constructor
CatalogStandardNodeMetadata::CatalogStandardNodeMetadata(
    const CatalogStandardNodeMetadata &pdbNodeToCopy) {
  nodeId = pdbNodeToCopy.nodeId;
  nodeIP = pdbNodeToCopy.nodeIP;
  nodePort = pdbNodeToCopy.nodePort;
  nodeName = pdbNodeToCopy.nodeName;
  nodeType = pdbNodeToCopy.nodeType;
  nodeStatus = pdbNodeToCopy.nodeStatus;
  nodeAddress =
      std::string(nodeIP) + std::string(":") + std::to_string(nodePort);
}

CatalogStandardNodeMetadata::~CatalogStandardNodeMetadata() {}

void CatalogStandardNodeMetadata::setValues(
    string nodeIdValue, string nodeIPValue, int nodePortValue,
    string nodeNameValue, string nodeTypeValue, int nodeStatusValue) {
  nodeId = nodeIPValue;
  nodeIP = nodeIPValue;
  nodePort = nodePortValue;
  nodeName = nodeNameValue;
  nodeType = nodeTypeValue;
  nodeStatus = nodeStatusValue;
  nodeAddress =
      std::string(nodeIP) + std::string(":") + std::to_string(nodePort);
}

string CatalogStandardNodeMetadata::getItemKey() { return nodeIP; }

string CatalogStandardNodeMetadata::getNodeIP() { return nodeIP; }

string CatalogStandardNodeMetadata::getItemId() { return nodeId; }

string CatalogStandardNodeMetadata::getItemName() { return nodeName; }

string CatalogStandardNodeMetadata::getNodeType() { return nodeType; }

int CatalogStandardNodeMetadata::getNodePort() { return nodePort; }

int CatalogStandardNodeMetadata::getNodeStatus() { return nodeStatus; }

void CatalogStandardNodeMetadata::setItemKey(string &itemKeyIn) {
  nodeAddress = itemKeyIn;
}

void CatalogStandardNodeMetadata::setItemId(string &itemIdIn) {
  nodeId = itemIdIn;
}

void CatalogStandardNodeMetadata::setItemIP(pdb::String &itemIPIn) {
  nodeIP = itemIPIn;
}

void CatalogStandardNodeMetadata::setItemName(string &itemNameIn) {
  nodeName = itemNameIn;
}

void CatalogStandardNodeMetadata::setNodePort(int &portIn) {
  nodePort = portIn;
}

string CatalogStandardNodeMetadata::printShort() {
  string output;
  output = "   Node address: ";
  output.append(getNodeIP().c_str())
      .append(" | ")
      .append(getItemName().c_str())
      .append(" | ")
      .append(getNodeType().c_str());
  return output;
}

void CatalogStandardNodeMetadata::toStandardMetadata(
    pdb::Handle<pdb::CatalogNodeMetadata> &convertedItem) {
  //        convertedItem->setValues();
}

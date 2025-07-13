#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <iomanip>
#include <NvInfer.h>
#include <NvInferRuntime.h>

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << msg << std::endl;
        }
    }
};

std::string dimensionsToString(const nvinfer1::Dims& dims) {
    std::string result = "[";
    for (int i = 0; i < dims.nbDims; ++i) {
        if (i > 0) result += ", ";
        if (dims.d[i] == -1) {
            result += "dynamic";
        } else {
            result += std::to_string(dims.d[i]);
        }
    }
    result += "]";
    return result;
}

std::string dataTypeToString(nvinfer1::DataType dataType) {
    switch (dataType) {
        case nvinfer1::DataType::kFLOAT: return "FLOAT32";
        case nvinfer1::DataType::kHALF: return "FLOAT16";
        case nvinfer1::DataType::kINT8: return "INT8";
        case nvinfer1::DataType::kINT32: return "INT32";
        case nvinfer1::DataType::kBOOL: return "BOOL";
        case nvinfer1::DataType::kUINT8: return "UINT8";
        case nvinfer1::DataType::kFP8: return "FP8";
        case nvinfer1::DataType::kBF16: return "BFLOAT16";
        case nvinfer1::DataType::kINT64: return "INT64";
        case nvinfer1::DataType::kINT4: return "INT4";
        default: return "UNKNOWN";
    }
}

std::string formatToString(nvinfer1::TensorFormat format) {
    switch (format) {
        case nvinfer1::TensorFormat::kLINEAR: return "LINEAR";
        case nvinfer1::TensorFormat::kCHW2: return "CHW2";
        case nvinfer1::TensorFormat::kCHW4: return "CHW4";
        case nvinfer1::TensorFormat::kCHW16: return "CHW16";
        case nvinfer1::TensorFormat::kCHW32: return "CHW32";
        case nvinfer1::TensorFormat::kDHWC8: return "DHWC8";
        case nvinfer1::TensorFormat::kCDHW32: return "CDHW32";
        case nvinfer1::TensorFormat::kHWC: return "HWC";
        case nvinfer1::TensorFormat::kDLA_LINEAR: return "DLA_LINEAR";
        case nvinfer1::TensorFormat::kDLA_HWC4: return "DLA_HWC4";
        case nvinfer1::TensorFormat::kHWC16: return "HWC16";
        case nvinfer1::TensorFormat::kDHWC: return "DHWC";
        default: return "UNKNOWN";
    }
}

bool loadEngine(const std::string& enginePath, std::unique_ptr<nvinfer1::ICudaEngine>& engine) {
    Logger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
    if (!runtime) {
        std::cerr << "Failed to create TensorRT runtime" << std::endl;
        return false;
    }

    std::ifstream engineFile(enginePath, std::ios::binary);
    if (!engineFile) {
        std::cerr << "Failed to open engine file: " << enginePath << std::endl;
        return false;
    }

    engineFile.seekg(0, std::ios::end);
    size_t engineSize = engineFile.tellg();
    engineFile.seekg(0, std::ios::beg);

    std::vector<char> engineData(engineSize);
    engineFile.read(engineData.data(), engineSize);
    engineFile.close();

    engine.reset(runtime->deserializeCudaEngine(engineData.data(), engineSize));
    if (!engine) {
        std::cerr << "Failed to deserialize engine" << std::endl;
        return false;
    }

    return true;
}

void printTableHeader() {
    std::cout << std::string(120, '=') << std::endl;
    std::cout << std::left 
              << std::setw(8) << "Index"
              << std::setw(12) << "Type"
              << std::setw(20) << "Name"
              << std::setw(12) << "Data Type"
              << std::setw(20) << "Min Shape"
              << std::setw(20) << "Opt Shape"
              << std::setw(20) << "Max Shape"
              << std::setw(8) << "Format" << std::endl;
    std::cout << std::string(120, '-') << std::endl;
}

void printTableRow(int index, const std::string& type, const std::string& name, 
                   const std::string& dataType, const std::string& minShape,
                   const std::string& optShape, const std::string& maxShape,
                   const std::string& format) {
    std::cout << std::left
              << std::setw(8) << index
              << std::setw(12) << type
              << std::setw(20) << (name.length() > 19 ? name.substr(0, 16) + "..." : name)
              << std::setw(12) << dataType
              << std::setw(20) << (minShape.length() > 19 ? minShape.substr(0, 16) + "..." : minShape)
              << std::setw(20) << (optShape.length() > 19 ? optShape.substr(0, 16) + "..." : optShape)
              << std::setw(20) << (maxShape.length() > 19 ? maxShape.substr(0, 16) + "..." : maxShape)
              << std::setw(8) << format << std::endl;
}

void printEngineInfo(nvinfer1::ICudaEngine* engine) {
    std::cout << "\n=== TensorRT Engine Information ===" << std::endl;
    std::cout << "Engine name: " << engine->getName() << std::endl;
    std::cout << "Number of I/O tensors: " << engine->getNbIOTensors() << std::endl;
    std::cout << "Number of optimization profiles: " << engine->getNbOptimizationProfiles() << std::endl;
    std::cout << std::endl;

    printTableHeader();

    for (int i = 0; i < engine->getNbIOTensors(); ++i) {
        const char* name = engine->getIOTensorName(i);
        nvinfer1::TensorIOMode ioMode = engine->getTensorIOMode(name);
        std::string type = (ioMode == nvinfer1::TensorIOMode::kINPUT) ? "INPUT" : "OUTPUT";
        std::string dataType = dataTypeToString(engine->getTensorDataType(name));
        auto format = engine->getTensorFormat(name);
        std::string formatStr = formatToString(format);
        
        // For the main profile (profile 0)
        auto minDims = engine->getProfileShape(name, 0, nvinfer1::OptProfileSelector::kMIN);
        auto optDims = engine->getProfileShape(name, 0, nvinfer1::OptProfileSelector::kOPT);
        auto maxDims = engine->getProfileShape(name, 0, nvinfer1::OptProfileSelector::kMAX);
        
        std::string minShape = dimensionsToString(minDims);
        std::string optShape = dimensionsToString(optDims);
        std::string maxShape = dimensionsToString(maxDims);
        
        printTableRow(i, type, name, dataType, minShape, optShape, maxShape, formatStr);
        
        // If there are multiple profiles, show them as additional rows
        for (int profile = 1; profile < engine->getNbOptimizationProfiles(); ++profile) {
            auto minDims_p = engine->getProfileShape(name, profile, nvinfer1::OptProfileSelector::kMIN);
            auto optDims_p = engine->getProfileShape(name, profile, nvinfer1::OptProfileSelector::kOPT);
            auto maxDims_p = engine->getProfileShape(name, profile, nvinfer1::OptProfileSelector::kMAX);
            
            std::string minShape_p = dimensionsToString(minDims_p);
            std::string optShape_p = dimensionsToString(optDims_p);
            std::string maxShape_p = dimensionsToString(maxDims_p);
            
            printTableRow(i, "Profile " + std::to_string(profile), "", "", minShape_p, optShape_p, maxShape_p, "");
        }
        
        // Show shape tensor info if applicable
        if (engine->isShapeInferenceIO(name)) {
            printTableRow(i, "Note", "Shape tensor", "", "", "", "", "");
        }
    }
    
    std::cout << std::string(120, '=') << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <engine_file>" << std::endl;
        return 1;
    }

    std::string enginePath = argv[1];
    std::unique_ptr<nvinfer1::ICudaEngine> engine;

    if (!loadEngine(enginePath, engine)) {
        return 1;
    }

    printEngineInfo(engine.get());

    return 0;
}
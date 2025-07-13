#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <onnxruntime/onnxruntime_cxx_api.h>

std::string typeToString(ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return "float32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return "uint8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: return "int8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: return "uint16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: return "int16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return "int32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return "int64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING: return "string";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return "bool";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return "float16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return "float64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: return "uint32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: return "uint64";
        default: return "unknown";
    }
}

std::string shapeToString(const std::vector<int64_t>& shape) {
    if (shape.empty()) return "[]";
    
    std::string result = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) result += ", ";
        if (shape[i] == -1) {
            result += "None";
        } else {
            result += std::to_string(shape[i]);
        }
    }
    result += "]";
    return result;
}

void printTableHeader() {
    std::cout << std::string(120, '=') << std::endl;
    std::cout << std::left 
              << std::setw(8) << "Index"
              << std::setw(12) << "Type"
              << std::setw(20) << "Name"
              << std::setw(12) << "Data Type"
              << std::setw(20) << "Shape"
              << std::setw(20) << ""
              << std::setw(20) << ""
              << std::setw(8) << "" << std::endl;
    std::cout << std::string(120, '-') << std::endl;
}

void printTableRow(int index, const std::string& type, const std::string& name, 
                   const std::string& dataType, const std::string& shape) {
    std::cout << std::left
              << std::setw(8) << index
              << std::setw(12) << type
              << std::setw(20) << (name.length() > 19 ? name.substr(0, 16) + "..." : name)
              << std::setw(12) << dataType
              << std::setw(20) << (shape.length() > 19 ? shape.substr(0, 16) + "..." : shape)
              << std::setw(20) << ""
              << std::setw(20) << ""
              << std::setw(8) << "" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " model_path" << std::endl;
        return -1;
    }
    
    std::string model_path = argv[1];
    
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnxinfo");
        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        Ort::Session session(env, model_path.c_str(), session_options);
        Ort::AllocatorWithDefaultOptions allocator;
        
        std::cout << "\n=== ONNX Model Information ===" << std::endl;
        std::cout << "Model path: " << model_path << std::endl;
        std::cout << "Number of inputs: " << session.GetInputCount() << std::endl;
        std::cout << "Number of outputs: " << session.GetOutputCount() << std::endl;
        std::cout << std::endl;
        
        printTableHeader();
        
        size_t input_count = session.GetInputCount();
        for (size_t i = 0; i < input_count; ++i) {
            auto input_name = session.GetInputNameAllocated(i, allocator);
            auto input_type_info = session.GetInputTypeInfo(i);
            auto tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
            
            std::string name = input_name.get();
            std::string type = typeToString(tensor_info.GetElementType());
            std::vector<int64_t> shape = tensor_info.GetShape();
            
            printTableRow(i, "INPUT", name, type, shapeToString(shape));
        }
        
        size_t output_count = session.GetOutputCount();
        for (size_t i = 0; i < output_count; ++i) {
            auto output_name = session.GetOutputNameAllocated(i, allocator);
            auto output_type_info = session.GetOutputTypeInfo(i);
            auto tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
            
            std::string name = output_name.get();
            std::string type = typeToString(tensor_info.GetElementType());
            std::vector<int64_t> shape = tensor_info.GetShape();
            
            printTableRow(input_count + i, "OUTPUT", name, type, shapeToString(shape));
        }
        
        std::cout << std::string(120, '=') << std::endl;
        
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}

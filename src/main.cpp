/*
# ID   : 胖胖的鱼
# @Time    : 2023/1/6 14:28
*/
#pragma once
#include <iostream>
#include <fstream>
#include <chrono>
#include<cmath>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvInferRuntime.h>
#include<opencv2/opencv.hpp>

//项目
#include "logging.h"

using namespace nvinfer1;
using namespace std;


#define API __declspec(dllexport)
#define CUDA_CHECK(callstr)\
    {\
        cudaError_t error_code = callstr;\
        if (error_code != cudaSuccess) {\
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__;\
            assert(0);\
        }\
    }


static Logger gLogger;
static int img_size;

struct alignas(float) BBOX{
	//float bbox[4];  //坐标
	cv::Rect_<float> box;
	float confidence;  // bbox_conf * cls_conf        坐标回归的置信度 * 类别的置信度
	float index; //类别
};

typedef struct{
	float* intput;  //输入指针
	float* out;     //输出指针

	//cuda实例化
	IRuntime* runtime;
	ICudaEngine* engine;
	IExecutionContext* context;
	cudaStream_t stream;

	int NUM_CLASSES, Max_anchors;    //类别数量，anchors数量

	static const int in_out_Bindings = 5;   //输入输出数量
	void* buffers[in_out_Bindings];         //输入输出层指针
	std::vector<int64_t> buffer_size;    //创建的大小

	cv::Mat img;                //图片
	vector<int> Classes;        //类别容器
	vector<cv::Rect> boxes;     //坐标容器
	vector<float> confidences;  //置信度容器
	vector<int> indices;        //所有目标信息

}YOLO;

//加载engine
bool load_engien(char*& trtModelStream, size_t& size, char* model_path){
	string engine_name = model_path;
	// 反序列化 .engine 并运行推理
	std::ifstream file(engine_name, std::ios::binary);      //engine路径  ofstream是从内存到硬盘，ifstream是从硬盘到内存，其实所谓的流缓冲就是内存空间;
	if (!file.good()) {  //判断当前流的状态（读写正常（即符合读取和写入的类型)，没有文件末尾）
		std::cerr << "[Error]: Read {" << engine_name << "} Error！" << std::endl;
		return false;
	}
	file.seekg(0, file.end);            //seekg()是对输入流的操作,从文件末尾开始计算偏移量
	size = file.tellg();                //得到file的大小
	file.seekg(0, file.beg);            //从文件头开始计算偏移量
	trtModelStream = new char[size];    //创建一个堆区
	file.read(trtModelStream, size);    //读取trtModelStream的size字节
	file.close();                       //关闭
	return true;
}

//计算dims
int volume(nvinfer1::Dims dims){
	int temp = 1;
	for (int i = 0; i < dims.nbDims; i++){
		temp *= dims.d[i];
	}
	return temp;
}

//创建显存
void create_runtime(char*& trtModelStream, size_t& size, YOLO*& trt){
	trt->runtime = nvinfer1::createInferRuntime(gLogger);
	trt->engine = trt->runtime->deserializeCudaEngine(trtModelStream, size);
	trt->context = trt->engine->createExecutionContext();
	delete[] trtModelStream;
	trt->NUM_CLASSES = trt->engine->getBindingDimensions(trt->engine->getNbBindings() - 1).d[2] - 5;
	trt->Max_anchors = trt->engine->getBindingDimensions(trt->engine->getNbBindings() - 1).d[1];
	img_size = trt->engine->getBindingDimensions(0).d[2];
	trt->buffer_size.resize(trt->engine->getNbBindings());    //初始化

	for (int i = 0; i < trt->engine->getNbBindings(); i++) {  //循环创建
		nvinfer1::Dims dims = trt->engine->getBindingDimensions(i);      //维度
		nvinfer1::DataType dtype = trt->engine->getBindingDataType(i);   //类型

		//int64_t total_size = volume(dims) * sizeof(dtype);  //sizeof(dtype) = 4
		int64_t total_size = static_cast<unsigned long long>(1) * volume(dims) * sizeof(dtype);  //sizeof(dtype) = 4
		trt->buffer_size[i] = total_size;
		CUDA_CHECK(cudaMalloc(&trt->buffers[i], total_size));    //创建显存
	}
	CUDA_CHECK(cudaStreamCreate(&trt->stream));
	//PC内存
	trt->intput = new float[trt->engine->getBindingDimensions(0).d[1] * trt->engine->getBindingDimensions(0).d[2] * trt->engine->getBindingDimensions(0).d[3]];
	trt->out = new float[trt->engine->getBindingDimensions(trt->engine->getNbBindings() - 1).d[0] * trt->engine->getBindingDimensions(trt->engine->getNbBindings() - 1).d[1] * trt->engine->getBindingDimensions(trt->engine->getNbBindings() - 1).d[2]];
}

//预处理
void preprocessing(cv::Mat& img, YOLO*& trt){
	cv::resize(img, img, cv::Size(trt->engine->getBindingDimensions(0).d[2], trt->engine->getBindingDimensions(0).d[3]));   //裁减640 0ms
	for (int c = 0; c < 3; ++c) { //1-2 3ms ，最低
		for (int h = 0; h < img.rows; ++h){
			//获取第i行首像素指针 
			cv::Vec3b* p1 = img.ptr<cv::Vec3b>(h);
			for (int w = 0; w < img.cols; ++w){
				trt->intput[c * img.cols * img.rows + h * img.cols + w] = (p1[w][c]) / 255.0f;
			}
		}
	}
}

//推理
void Inference(YOLO*& trt){
	CUDA_CHECK(cudaMemcpyAsync(trt->buffers[0], trt->intput, trt->buffer_size[0], cudaMemcpyHostToDevice, trt->stream));
	//context->enqueue(1, buffers, stream, nullptr);    //异步
	trt->context->executeV2(trt->buffers);    //同步
	CUDA_CHECK(cudaMemcpyAsync(trt->out, trt->buffers[trt->engine->getNbBindings() - 1], trt->buffer_size[trt->engine->getNbBindings() - 1], cudaMemcpyDeviceToHost, trt->stream));
	//cudaStreamSynchronize(stream);    //
}

//onnx转engine
void OnnxToEngine(IHostMemory** modelStream, std::string& onnx_name, std::string& engine_name, int& HasFast){
	IBuilder* builder = createInferBuilder(gLogger);
	INetworkDefinition* network = builder->createNetworkV2(1);
	auto parser = nvonnxparser::createParser(*network, gLogger);				//cmake缺少链接库 ${TARGET_NAME}nvonnxparser
	parser->parseFromFile(onnx_name.c_str(), (int)nvinfer1::ILogger::Severity::kWARNING);
	IBuilderConfig* config = builder->createBuilderConfig();
	config->setMaxWorkspaceSize(static_cast<size_t>(16) * (static_cast<unsigned long long>(1) << 20));  
	if (builder->platformHasFastFp16() && HasFast == 0){
		cout << "[INFO]: is FP16" << endl;
		config->setFlag(BuilderFlag::kFP16);    //设置为fp16
	}else cout << "[INFO]: is FP32" << endl;
	std::cout << "[INFO]: Builld engine, please wait 10-15 minutes..." << endl;
	nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
	(*modelStream) = engine->serialize();

	std::ofstream p(engine_name, std::ios::binary);
	if (p) {
		p.write(reinterpret_cast<const char*>((*modelStream)->data()), (*modelStream)->size());
		std::cout << "[INFO]: Save .engine file successfully  √" << std::endl;
	}
	else{
		std::cout << "[Error]: save failed ×" << std::endl;
	}

	// 释放
	engine->destroy();
	config->destroy();
	network->destroy();
	builder->destroy();
}

//显示GPU信息
int show_GPU(){
	cudaDeviceProp deviceProp;
	int deviceCount;
	cudaError_t cudaError;
	cudaError = cudaGetDeviceCount(&deviceCount);
	if (deviceCount == 0) {
		std::cout << "[Error]: No nvidia graphics card,Please update the driver " << std::endl;
		return -1;
	}
	return deviceCount;
}

extern "C" API int GetInputSize() {
	return img_size;
}

//初始化
extern "C" API void* Init(char* model_path, int Build_Flag){
	YOLO* trt = new YOLO();
	char* trtModelStream = nullptr;     //创建流
	size_t size = 0;
	load_engien(trtModelStream, size, model_path);  //加载模型

	//创建接口
	create_runtime(trtModelStream, size, trt);
	return (void*)trt;
}

static bool Is_File(const std::string& file_path) {
	std::ifstream file(file_path.c_str());
	return file.good();
}

//生成engine
extern "C" API void Build(char* onnx_path_python, char* engine_path_python, int HasFast,int Device=0){
	//显示PU信息
	int Device_num = show_GPU();
	if (Device_num == -1) {
		return;
	}

	if (Device_num < Device) {
		return;
	}

	std::string onnx_name = onnx_path_python;      //onnx路径
	std::string engine_name = engine_path_python;   //engine名字

	cudaSetDevice(Device);          //选择设备

	//检测engine文件
	if (Is_File(onnx_name)){
		IHostMemory* modelStream{ nullptr };   
		cout << "--------------------------------------- Onnx to Engine --------------------------------------" << endl;
		cout << "[INFO]: Onnx file path:" << onnx_name << endl;
		cout << "[INFO]: engine output path :" << engine_name << endl;
		cout << "-------------------------------------- Version: python --------------------------------------" << endl;
		OnnxToEngine(&modelStream, onnx_name, engine_name,HasFast);
	} else {
		cout << "[Error]: Onnx file not find:" << onnx_name << endl;
	}
	
}

//预处理 + 推理 + 后处理
extern "C" API void Detect(YOLO * init_trt, int rows, int cols, float conf, float iou, unsigned char* src_data, float(*res_array)[6]){
	YOLO* trt = (YOLO*)init_trt;
	cv::resize(cv::Mat(rows, cols, CV_8UC3, src_data), trt->img, cv::Size(trt->engine->getBindingDimensions(0).d[2], trt->engine->getBindingDimensions(0).d[3]));   //裁减640 0ms

	for (int c = 0; c < 3; ++c) {//1-2 3ms ，最低
		for (int h = 0; h < trt->img.rows; ++h){
			//获取第i行首像素指针 
			cv::Vec3b* p1 = trt->img.ptr<cv::Vec3b>(h);
			for (int w = 0; w < trt->img.cols; ++w){
				trt->intput[c * trt->img.cols * trt->img.rows + h * trt->img.cols + w] = (p1[w][c]) / 255.0f;
			}
		}
	}

	Inference(trt);

	trt->Classes.clear();
	trt->boxes.clear();
	trt->confidences.clear();
	trt->indices.clear();

	for (int i = 0; i < trt->Max_anchors; i++){
		//置信度
		float tempConf = *max_element(&trt->out[i * (5 + trt->NUM_CLASSES) + 5], &trt->out[(i + 1) * (5 + trt->NUM_CLASSES)]);
		if (tempConf < conf)
			continue;

		cv::Rect temp;
		temp.x = ((float*)trt->out)[i * (5 + trt->NUM_CLASSES)];
		temp.y = ((float*)trt->out)[i * (5 + trt->NUM_CLASSES) + 1];
		temp.width = ((float*)trt->out)[i * (5 + trt->NUM_CLASSES) + 2];
		temp.height = ((float*)trt->out)[i * (5 + trt->NUM_CLASSES) + 3];
		//类别
		int tempClass = max_element(&trt->out[i * (5 + trt->NUM_CLASSES) + 5], &trt->out[(i + 1) * (5 + trt->NUM_CLASSES)]) - &trt->out[i * (5 + trt->NUM_CLASSES) + 5];

		trt->Classes.push_back(tempClass);
		trt->boxes.push_back(temp);
		trt->confidences.push_back(((float*)trt->out)[i * (5 + trt->NUM_CLASSES) + 4] * tempConf);
	}

	//非极大值抑制
	cv::dnn::NMSBoxes(trt->boxes, trt->confidences, conf, iou, trt->indices);

	for (size_t i = 0; i < trt->indices.size(); i++){
		res_array[i][0] = trt->boxes[trt->indices[i]].x;
		res_array[i][1] = trt->boxes[trt->indices[i]].y;
		res_array[i][2] = trt->boxes[trt->indices[i]].width;
		res_array[i][3] = trt->boxes[trt->indices[i]].height;
		res_array[i][4] = trt->Classes[i];
		res_array[i][5] = trt->confidences[i];
	}
}

//释放
extern "C" API void Free(void* init_trt)
{
	YOLO* trt = (YOLO*)init_trt;
	delete[] trt->intput;
	delete[] trt->out;
	cudaStreamDestroy(trt->stream);
	for (int i = 0; i < trt->engine->getNbBindings(); i++)   {//循环释放
		CUDA_CHECK(cudaFree(trt->buffers[i]));
	}
	trt->context->destroy();
	trt->engine->destroy();
	trt->runtime->destroy();
}
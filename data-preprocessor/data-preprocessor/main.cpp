#include <fstream>
#include <vector>
#include <iostream>
#include "json.hpp"





constexpr size_t wordSize = 1;

//256 ^ wordSize
constexpr size_t dictSize = 256;
std::string ipath("../../dataset/samples/");
std::string opath("../../dataset/output-samples/");

namespace fs = std::filesystem;


std::vector<float> GetRelativeFrequency(char* buf, size_t len)
{
	/*
	std::vector<uint64_t> freqs;
	freqs.resize(dictSize);
	for (size_t i = 0; i < len; i+=wordSize)
	{
		char word[wordSize];
		memcpy()
	}
	*/
	std::vector<uint64_t> count;
	std::vector<float> freqs;
	freqs.resize(dictSize);
	count.resize(dictSize);

	for (size_t i = 0; i < len; i += wordSize)
	{
		unsigned char cchar = buf[i];
		//std::cout << std::hex << (int)cchar << std::endl;;
		count [cchar] += 1;
	}
	
	
	for (size_t i = 0; i < dictSize; i++)
	{
		freqs[i] = (double)count[i] / (double)len;
	}
	return freqs;

}


size_t GetPaddedSize(size_t realSize)
{
	size_t rem = realSize % wordSize;
	if (rem == 0)
		return realSize;
	return realSize + (wordSize - rem);
}


void ParseData(fs::directory_entry in_entry)
{

	fs::path in_path = in_entry.path();
	using namespace std;
	using namespace nlohmann;
	ifstream ifs;
	ofstream ofs;
	size_t bufLen = GetPaddedSize(in_entry.file_size());
	char* buf = new char[bufLen](0);

	ifs.open(in_path, ios::binary);
	ifs.read(buf, in_entry.file_size());


	std::vector<float> freqs = GetRelativeFrequency(buf, bufLen);
	json j(freqs);
	std::string out = j.dump();
	std::string outpath = opath + in_path.filename().string() + ".json";
	ofs.open(outpath);
	ofs << out << std::endl;
	ofs.close();



	
}



int main()
{
	

	for (const auto& entry : fs::directory_iterator(ipath))
		ParseData(entry);
}
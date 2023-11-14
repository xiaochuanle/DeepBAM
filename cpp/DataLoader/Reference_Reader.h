//
// Created by dell on 2023/8/11.
//

#ifndef BAMCLASS_REFERENCE_READER_H
#define BAMCLASS_REFERENCE_READER_H
#include <map>
#include <string>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
namespace fs = std::filesystem;
namespace Yao {
    class Reference_Reader {
    public:
        Reference_Reader(fs::path reference_path_, std::string seq_type_);

        Reference_Reader();

        std::string get_reference_seq(std::string chr_key,
                                      bool is_forward,
                                      int64_t start,
                                      int64_t end);
        void display() {
            for (const auto &[key, value] : chr_to_ref) {
                std::cout << key << " " << value.length() << std::endl;
            }
        }

    private:
        fs::path reference_path;
        std::map<std::string, std::string> chr_to_ref;
        std::set<std::string> chr_set;
        std::string seq_type;
    };
}
#endif //BAMCLASS_REFERENCE_READER_H

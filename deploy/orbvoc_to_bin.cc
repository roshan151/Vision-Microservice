// ORBvoc.txt -> ORBvoc.bin for ORB-SLAM3 patched with deploy/orbslam3_binary_vocabulary.patch.
// Loads the text vocabulary, writes the binary one, reads that back and checks it is the same tree
// (every node's parent, children, word, weight and descriptor bytes) before keeping it.
// Built and run by deploy/install_orbslam3_macos.sh.
//
//   orbvoc_to_bin ORBvoc.txt ORBvoc.bin
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>

#include "Thirdparty/DBoW2/DBoW2/FORB.h"
#include "Thirdparty/DBoW2/DBoW2/TemplatedVocabulary.h"

typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB> Vocab;

struct Peek : Vocab {          // the tree is protected
  static std::string diff(const Peek &a, const Peek &b) {
    if (a.m_k != b.m_k || a.m_L != b.m_L || a.m_scoring != b.m_scoring || a.m_weighting != b.m_weighting)
      return "header";
    if (a.m_nodes.size() != b.m_nodes.size()) return "node count";
    if (a.m_words.size() != b.m_words.size()) return "word count";
    for (size_t i = 0; i < a.m_nodes.size(); ++i) {
      const Node &x = a.m_nodes[i], &y = b.m_nodes[i];
      if (x.id != y.id || x.parent != y.parent || x.children != y.children || x.word_id != y.word_id ||
          std::memcmp(&x.weight, &y.weight, sizeof(x.weight)) != 0)
        return "node " + std::to_string(i);
      if (i > 0 && (x.descriptor.total() != y.descriptor.total() || x.descriptor.type() != y.descriptor.type() ||
                    std::memcmp(x.descriptor.data, y.descriptor.data, x.descriptor.total() * x.descriptor.elemSize()) != 0))
        return "descriptor of node " + std::to_string(i);
    }
    for (size_t w = 0; w < a.m_words.size(); ++w)
      if (a.m_words[w] - a.m_nodes.data() != b.m_words[w] - b.m_nodes.data()) return "word " + std::to_string(w);
    return "";
  }
};

static double since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

int main(int argc, char **argv) {
  if (argc != 3) { std::cerr << "usage: " << argv[0] << " ORBvoc.txt ORBvoc.bin\n"; return 2; }
  const std::string txt = argv[1], bin = argv[2], tmp = bin + ".tmp";
  Peek a, b;
  auto t0 = std::chrono::steady_clock::now();
  if (!a.loadFromTextFile(txt)) { std::cerr << "could not load " << txt << "\n"; return 1; }
  std::cout << "text vocabulary: " << a.size() << " words, loaded in " << since(t0) << " s\n";
  if (!a.saveToBinaryFile(tmp)) { std::cerr << "could not write " << tmp << "\n"; return 1; }
  t0 = std::chrono::steady_clock::now();
  if (!b.loadFromBinaryFile(tmp)) { std::cerr << "could not read back " << tmp << "\n"; std::remove(tmp.c_str()); return 1; }
  std::cout << "binary vocabulary: loaded in " << since(t0) << " s\n";
  const std::string d = Peek::diff(a, b);
  if (!d.empty()) { std::cerr << "binary vocabulary differs from the text one (" << d << ")\n"; std::remove(tmp.c_str()); return 1; }
  if (std::rename(tmp.c_str(), bin.c_str()) != 0) { std::cerr << "could not rename " << tmp << "\n"; return 1; }
  std::cout << "wrote " << bin << " (identical tree)\n";
  return 0;
}

// ANTLR
#include "VCalcLexer.h"
#include "VCalcParser.h"

#include "ANTLRFileStream.h"
#include "CommonTokenStream.h"
#include "tree/ParseTree.h"
#include "tree/ParseTreeWalker.h"

// Our backend
#include "BackEnd.h"

// Command line from llvm::cl
#include "llvm/Support/CommandLine.h"

// Standard
#include <iostream>
#include <fstream>
#include <string>

namespace cl = llvm::cl;

cl::opt<std::string> OutputFilename("o", cl::desc("Specify the output filename"), cl::value_desc("filename"));
cl::opt<std::string> InputFilename(cl::Positional, cl::desc("<input file>"), cl::Required); //cl::init("-"));

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv);

  // Get our outfile
  // std::string outfile(OutputFilename.c_str());
  std::string outfile = "";
  if (outfile == "") outfile = "a.out";
  std::ofstream out(outfile, std::ios::binary);

  // get the input file
  // std::string infile(InputFilename.c_str());
  std::string infile = "test.in";

  // Open the file then parse and lex it.
  antlr4::ANTLRFileStream afs;
  afs.loadFromFile(infile);
  vcalc::VCalcLexer lexer(&afs);
  antlr4::CommonTokenStream tokens(&lexer);
  vcalc::VCalcParser parser(&tokens);

  // Get the root of the parse tree. Use your base rule name.
  antlr4::tree::ParseTree *tree = parser.file();

  // HOW TO USE A VISITOR
  // Make the visitor
  // MyVisitor visitor;
  // Visit the tree
  // visitor.visit(tree);

  std::ofstream os(argv[2]);
  BackEnd backend;
  backend.emitModule();
  backend.lowerDialects();
  backend.dumpLLVM(os);

  return 0;
}

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <tuple>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/date_time.hpp>
#include <boost/process.hpp>
#include <boost/lexical_cast.hpp>

using namespace std;
using namespace boost;
using namespace boost::filesystem;
using namespace boost::program_options;
using namespace boost::posix_time;
using namespace boost::process;

int main(int argc, char ** argv)
{
  string input;
  string output;
  options_description desc;
  desc.add_options()
    ("help,h", "print current message")
    ("input,i", value<string>(&input), "input file")
    ("output,o", value<string>(&output), "output file");
  variables_map vm;
  store(parse_command_line(argc, argv, desc), vm);
  notify(vm);
  // check arguments
  if (0 != vm.count("help")) {
    cout<<desc<<endl;
    return EXIT_SUCCESS;
  }
  if (1 != vm.count("input") || 1 != vm.count("output")) {
    cerr<<"single input or output must be given!"<<endl;
    return EXIT_FAILURE;
  }
  if (false == exists(input) || true == is_directory(input)) {
    cerr<<"invalid input file"<<endl;
    return EXIT_FAILURE;
  }
  if (true == exists(output)) {
    cerr<<"output location is occupied, will not overwrite"<<endl;
    return EXIT_FAILURE;
  }
  // convert input from utf16 to utf8
  string line;
  std::ofstream tmp("tmp");
  ipstream buffer;
  child c(search_path("iconv"), vector<string>{"-f", "UTF-16LE", "-t", "UTF-8", input}, std_out > buffer);
  while(getline(buffer, line)) {
    tmp << line << endl;
  }
  tmp.close();
  c.wait();
  // read from tmp file
  map<ptime, std::tuple<float,float> > ticks;
  bool first_line = true;
  std::ifstream in("tmp");
  while(getline(in, line)) {
    // skip table header
    if (first_line) {
      first_line = false;
      continue;
    }
    trim(line);
    if (line == "") continue;
    vector<string> tokens;
    split(tokens, line, is_any_of("\t"));
    // parse the tokens
    ptime tick_time = time_from_string(tokens[0]);
    float sell_price = lexical_cast<float>(tokens[1]);
    float buy_price = lexical_cast<float>(tokens[2]);
    // always use the latest tick at every second
    ticks[tick_time] = std::make_tuple(sell_price, buy_price);
  }
  in.close();
  remove_all("tmp");
  // write to output file
  std::ofstream out(output);
  for (auto& tick : ticks) {
    out << tick.first << "\t" << std::get<0>(tick.second) << "\t" << std::get<1>(tick.second) << endl;
  }
  out.close();
  return EXIT_SUCCESS;
}


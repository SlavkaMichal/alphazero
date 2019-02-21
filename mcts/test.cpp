#include <array>
#include <algorithm>
#include <valarray>
#include <type_traits>
#include <typeinfo>
#ifndef _MSC_VER
#   include <cxxabi.h>
#endif
#include <memory>
#include <cstdlib>
#include <iostream>
#include <ostream>
#include <string>
template <class T>
std::string
type_name()
{
    typedef typename std::remove_reference<T>::type TR;
    std::unique_ptr<char, void(*)(void*)> own
           (
#ifndef _MSC_VER
                abi::__cxa_demangle(typeid(TR).name(), nullptr,
                                           nullptr, nullptr),
#else
                nullptr,
#endif
                std::free
           );
    std::string r = own != nullptr ? own.get() : typeid(TR).name();
    if (std::is_const<TR>::value)
        r += " const";
    if (std::is_volatile<TR>::value)
        r += " volatile";
    if (std::is_lvalue_reference<T>::value)
        r += "&";
    else if (std::is_rvalue_reference<T>::value)
        r += "&&";
    return r;
}

using arr = std::array<std::array<float, 5>,5>;
int main(){
	arr a{};
	arr b{};
	std::array<float, 10> c{};
	float *ptr;

	for (int i = 0; i < 10; i++)
		c[i] = i;
	std::swap_ranges(c.begin(),c.begin()+5, c.begin()+5);
	std::cout << "after swap" <<std::endl;
	for (int i = 0; i < 10; i++)
		std::cout << i << ": " << c[i] << std::endl;
	std::cout << "------------" <<std::endl;
	a[0][3] = 6;
	a[1][2] = 7;
	std::cout << " array: " << sizeof a << std::endl << " element: " << sizeof a[1] << std::endl << " data: " << sizeof *a.data() << std::endl<< sizeof(float) << std::endl;
	std::cout << " array: " << type_name<decltype(a)>() << std::endl << " element: " << type_name<decltype(a[1])>() << std::endl << " data: " <<  type_name<decltype(a.data())>() << std::endl;

	std::cout << "before" <<std::endl;
	for (int i=0; i<5;i++){
		for (int j=0; j<5;j++)
			std::cout << " " << b[i][j];
		std::cout << std::endl;
	}

	std::cout << "after" <<std::endl;
	b = a;
	for (int i=0; i<5;i++){
		for (int j=0; j<5;j++)
			std::cout << " " << b[i][j];
		std::cout << std::endl;
	}
	std::cout << "after2" <<std::endl;
	b[0].swap(b[1]);
	for (int i=0; i<5;i++){
		for (int j=0; j<5;j++)
			std::cout << " " << b[i][j];
		std::cout << std::endl;
	}

	int i;
	i = 7;
	std::cout << "move " << i<<"y: "<<i/3 <<"x: "<< i%3 << std::endl;
	return 0;
}

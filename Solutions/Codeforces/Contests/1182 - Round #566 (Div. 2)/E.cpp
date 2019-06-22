#include <bits/stdc++.h>
using namespace std;

#define vec vector
#define FOR(i, a, b) for (int i = (a); i < (b); i++)

// Fermat's little theorem:
//  a^(p-1) = 1 (mod p) when p is a prime number.

// Constants
typedef long long ll;
const ll R = 1e9 + 7; // a^(R-1) = 1 (mod R)
const ll MATRIX_REMAINDER = R - 1;

// Matrix class
class Matrix {
public:
  // Attributes
  int row, col;
  vec<vec<ll> > num;
  
  // Constructors
  Matrix (int row, int col, int defaultValue = 0) {
    this->num = vec<vec<ll> >(row, vec<ll>(col, defaultValue));
    this->row = row, this->col = col;
  }
  Matrix (vec<vec<ll> > num) {
    this->num = num;
    this->row = this->num.size();
    this->col = this->num[0].size();
  }
  
  // Multiplication
  Matrix operator *(Matrix &other) {
    if (this->col != other.row) {
      printf("Wrong size: %d*%d X %d*%d\n", 
             this->row, this->col, other.row, other.col);
      throw "Wrong size";
    }
    Matrix res(this->row, other.col);
    FOR(r, 0, res.row) FOR(c, 0, res.col) {
      FOR(k, 0, this->col) {
        (res.num[r][c] += this->num[r][k] * other.num[k][c]) %= MATRIX_REMAINDER;
      }
    }
    return res;
  }
  
  // Power
  Matrix operator ^(ll x) {
    if (x == 0) {
      printf("Not implemented yet.\n");
      throw "Not implemented";
    }
    if (x == 1) {
      return *this;
    }
    Matrix res = (*this) ^ (x/2);
    res = res * res;
    if (x % 2){
      res = res * (*this);
    }
    return res;
  }
};

// Prime related
const int limit = 1 << 20;
bool notPrime[limit];
vec<ll> primes;

// Decompose given value to primes
vec<ll> primeDecomposition(ll x) {
  vec<ll> answer;
  for (ll p : primes) {
    if (x <= 1) {
      break;
    }
    if (x % p == 0) {
      answer.push_back(p);
      while (x % p == 0) {
        x /= p;
      }
    } else if (p * p > x) {
      answer.push_back(x);
      break;
    }
  }
  return answer;
}

// Return a^x % R
ll modPow(ll a, ll x) {
  if (x == 0) {
    return 1;
  }
  ll res = modPow(a, x / 2);
  (res *= res) %= R;
  if (x % 2) {
    (res *= a) %= R;
  }
  return res;
}

void sieve() {
  notPrime[0] = notPrime[1] = true;
  FOR(i, 2, limit) {
    if (!notPrime[i]) {
      primes.push_back(i);
      for (int j = i + i; j < limit; j += i) {
        notPrime[j] = true;
      }
    }
  }
}

void printMatrix(Matrix &b) {
  printf("  %lld %lld\n  %lld %lld\n",
         b.num[0][0], b.num[0][1],
         b.num[1][0], b.num[1][1]);
}

void testMatrix() {
  // Matrix functionality testing
  Matrix a1({{1, 2}, 
             {0, 1}});
  Matrix a2({{3, 2},
             {-1, 0}});
  Matrix b = a1 * a2;
  printf("a1*a2 = \n");
  printMatrix(b);
  // Expect:
  //    a1*a2 = 
  //      1 2
  //      -1 0
}

set<ll> getKnownPrimes(ll f[4], ll c) {
  // Calculate target primes
  set<ll> knownPrimes;
  FOR(i, 1, 4) {
    for (ll p : primeDecomposition(f[i])) {
      knownPrimes.insert(p);
    }
  }
  for (ll p : primeDecomposition(c)) {
    knownPrimes.insert(p);
  }
//   printf("Known primes: ");
//   for (ll p : knownPrimes) {
//     printf("%lld, ", p);
//   }
//   printf("\n");
  return knownPrimes;
}

map<ll, ll> getFnPrimeCount(
	set<ll> &knownPrimes, Matrix &totalLogPropagate, ll f[4], ll c
	) {
  map<ll, ll> fnPrimeCount;
  for (ll p : knownPrimes) {
    Matrix initPrimeCount(3, 1);
    FOR(i, 1, 4) {
      for (ll num = f[i]; num % p == 0; num /= p) {
        initPrimeCount.num[3-i][0]++;     // f[n]
      }
      for (ll num = c; num % p == 0; num /= p) {
        initPrimeCount.num[3-i][0] += i;  // c^n
      }
      Matrix lastPrimeCount = totalLogPropagate * initPrimeCount;
      fnPrimeCount[p] = lastPrimeCount.num[0][0];
    }
  }
  return fnPrimeCount;
}

// Main fucntion
int main() {
//   testMatrix();

  sieve();
 
  // Get input and init params
  ll n, f[4], c;
  scanf("%lld %lld %lld %lld %lld", &n, &f[1], &f[2], &f[3], &c);
  
  // Let g_n denote c^n * f_n. Then we have g_n = g_{n−1} * g_{n−2} * g_{n−3}, 
  // and thus for all n, g_n = (g_1)^{x_n} * (g_2)^{y_n} * (g_3)^{z_n}. Then the 
  // recurrence satisfied by x_n, y_n, z_n is x_n = x_{n−1} + x_{n−2} + x_{n−3}, 
  // and same for y, z. Now use matrix exponentiation modulo 10^9 + 6 to find the 
  // reduced exponents, and this enables us to find g_n using the formula above. 
  // Then we find c^{−n} * g_n using inverse of c_n, and thus we're done.
  
  // The required transformation matrix is given by the following three equations:
  //      x_n = 1 * x_{n-1} + 1 * x_{n-2} + 1 * x_{n-3}
  //  x_{n-1} = 1 * x_{n-1} + 0 * x_{n-2} + 0 * x_{n-3}
  //  x_{n-2} = 0 * x_{n-1} + 1 * x_{n-2} + 0 * x_{n-3}
  Matrix baseLogPropagate({{1, 1, 1},
                           {1, 0, 0},
                           {0, 1, 0}});
  Matrix totalLogPropagate = baseLogPropagate ^ (n - 3);
  
  set<ll> knownPrimes = getKnownPrimes(f, c);
  
  // Calculate c^n * f[n]'s prime count: initPrimeCount[x]: f[3-x]'s p-occurence
  map<ll, ll> fnPrimeCount = getFnPrimeCount(
  	knownPrimes, totalLogPropagate, f, c);
  	
  // Calculate answer = product(p^fnPrimeCount[p]) & c^(-n)
  ll answer = 1;
  for (auto pinfo : fnPrimeCount) {
//   	printf("%lld-occurence = %lld\n", pinfo.first, pinfo.second);
	(answer *= modPow(pinfo.first, pinfo.second)) %= R;
  }
  (answer *= modPow(modPow(c, R-2), n)) %= R;
  printf("%lld\n", answer);
  
  return 0;
}

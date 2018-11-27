Table of Content

0. Templete	1
1. Math	1
1. Bit manupulation	1
2. Extended Euclid	2
3. Calculate nCk	2
4. Gaussian Elimination	2
6. Freivalid	4
7. sqrt Bigdecimal	5
8. Mul/Power Matrix	5
9. Fibonanci	5
10. Chinese Reainder Theorem	6
11. Combination	7
2. Graph	7
1. Bridge & Articulation	7
2. Dijkstra + heap kc97ble	8
3. LCA kc97ble	8
4. Lehmer - Đếm số lượng số nguyên tố nhỏ hơn n	9
3. Geometry	10
1. Basic	10
2. ConvexHull	14
4. Data Structure	15
1. BIT 2D	15
2. Persistent Tree	17
5. String	19
1. KMP	19

## 0. Template
```c++
#include <bits/stdc++.h>

using namespace std;

typedef long long ll;
typedef vector<int> vi;
typedef vector< vector<int> > vvi;
typedef vector<ll> vl;
typedef vector< vector<ll> > vvl;
typedef pair<int, int> ii;
typedef vector<ii> vii;
typedef vector< vii> vvii;

#define FOR(i, a, b) \
   for (__typeof(b) i = (a); i < (b); i++)
#define REP(i, begin, end) \
   for (__typeof(end) i = (begin) - ((begin) > (end)); i != (end) - ((begin) > (end)); i += 1 - 2 * ((begin) > (end)))

#define PI 3.14159265

inline ll GCD(ll a, ll b) {while (b != 0) {ll c = a % b; a = b; b = c;} return a;};
inline ll LCM(ll a, ll b) {return (a / GCD(a,b)) * b;};

int main()
{
    ios_base::sync_with_stdio(false);
    cin.tie();
    freopen("", "r", stdin);
   cerr << "Hello" << endl;
   return 0;
}
```

## 1. Math 				
### 1. Bit manupulation
```
#define isOn(S, j) (S & (1 << j))
#define setBit(S, j) (S |= (1 << j))
#define clearBit(S, j) (S &= ~(1 << j))
#define toggleBit(S, j) (S ^= (1 << j))
#define lowBit(S) (S & (-S))
#define setAll(S, n) (S = (1 << n) - 1)

#define modulo(S, N) ((S) & (N - 1))   // returns S % N, where N is a power of 2
#define isPowerOfTwo(S) (!(S & (S - 1)))
#define nearestPowerOfTwo(S) ((int)pow(2.0, (int)((log((double)S) / log(2.0)) + 0.5)))
#define turnOffLastBit(S) ((S) & (S - 1))
#define turnOnLastZero(S) ((S) | (S + 1))
#define turnOffLastConsecutiveBits(S) ((S) & (S + 1))
#define turnOnLastConsecutiveZeroes(S) ((S) | (S - 1))
```
		
### 2. Extended Euclid
```
void extendedEuclid(ll a, ll b) { 
    if (b == 0) { x = 1; y = 0; d = a; return; }
    extendedEuclid(b, a % b);
    ll x1 = y;
    ll y1 = x - (a / b) * y;
    x = x1;
    y = y1;
}
```

3. Calculate nCk
ll C[nMAX][nMAX];
ll mod = 1e9+7;

void init() {
	C[0][0] = 1;
    REP(i,1,nMAX){
        C[i][0] = C[i][i] = 1;
        REP(j,1,i) C[i][j] = (C[i-1][j] + C[i-1][j-1]) % mod;
    }
}


		
4. Gaussian Elimination
// C++ program to demostrate working of Guassian Elimination 
// method 
#include<bits/stdc++.h> 
using namespace std; 
  
#define N 3        // Number of unknowns 
  
// function to reduce matrix to r.e.f.  Returns a value to  
// indicate whether matrix is singular or not 
int forwardElim(double mat[N][N+1]); 
  
// function to calculate the values of the unknowns 
void backSub(double mat[N][N+1]); 
  
// function to get matrix content 
void gaussianElimination(double mat[N][N+1]) 
{ 
    /* reduction into r.e.f. */
    int singular_flag = forwardElim(mat); 
  
    /* if matrix is singular */
    if (singular_flag != -1) 
    { 
        printf("Singular Matrix.\n"); 
  
        /* if the RHS of equation corresponding to 
           zero row  is 0, * system has infinitely 
           many solutions, else inconsistent*/
        if (mat[singular_flag][N]) 
            printf("Inconsistent System."); 
        else
            printf("May have infinitely many "
                   "solutions."); 
  
        return; 
    } 
  
    /* get solution to system and print it using 
       backward substitution */
    backSub(mat); 
} 
  
// function for elemntary operation of swapping two rows 
void swap_row(double mat[N][N+1], int i, int j) 
{ 
    //printf("Swapped rows %d and %d\n", i, j); 
  
    for (int k=0; k<=N; k++) 
    { 
        double temp = mat[i][k]; 
        mat[i][k] = mat[j][k]; 
        mat[j][k] = temp; 
    } 
} 
  
// function to print matrix content at any stage 
void print(double mat[N][N+1]) 
{ 
    for (int i=0; i<N; i++, printf("\n")) 
        for (int j=0; j<=N; j++) 
            printf("%lf ", mat[i][j]); 
  
    printf("\n"); 
} 
  
// function to reduce matrix to r.e.f. 
int forwardElim(double mat[N][N+1]) 
{ 
    for (int k=0; k<N; k++) 
    { 
        // Initialize maximum value and index for pivot 
        int i_max = k; 
        int v_max = mat[i_max][k]; 
  
        /* find greater amplitude for pivot if any */
        for (int i = k+1; i < N; i++) 
            if (abs(mat[i][k]) > v_max) 
                v_max = mat[i][k], i_max = i; 
  
        /* if a prinicipal diagonal element  is zero, 
         * it denotes that matrix is singular, and 
         * will lead to a division-by-zero later. */
        if (!mat[k][i_max]) 
            return k; // Matrix is singular 
  
        /* Swap the greatest value row with current row */
        if (i_max != k) 
            swap_row(mat, k, i_max); 
  
  
        for (int i=k+1; i<N; i++) 
        { 
            /* factor f to set current row kth elemnt to 0, 
             * and subsequently remaining kth column to 0 */
            double f = mat[i][k]/mat[k][k]; 
  
            /* subtract fth multiple of corresponding kth 
               row element*/
            for (int j=k+1; j<=N; j++) 
                mat[i][j] -= mat[k][j]*f; 
  
            /* filling lower triangular matrix with zeros*/
            mat[i][k] = 0; 
        } 
  
        //print(mat);        //for matrix state 
    } 
    //print(mat);            //for matrix state 
    return -1; 
} 
  
// function to calculate the values of the unknowns 
void backSub(double mat[N][N+1]) 
{ 
    double x[N];  // An array to store solution 
  
    /* Start calculating from last equation up to the 
       first */
    for (int i = N-1; i >= 0; i--) 
    { 
        /* start with the RHS of the equation */
        x[i] = mat[i][N]; 
  
        /* Initialize j to i+1 since matrix is upper 
           triangular*/
        for (int j=i+1; j<N; j++) 
        { 
            /* subtract all the lhs values 
             * except the coefficient of the variable 
             * whose value is being calculated */
            x[i] -= mat[i][j]*x[j]; 
        } 
  
        /* divide the RHS by the coefficient of the 
           unknown being calculated */
        x[i] = x[i]/mat[i][i]; 
    } 
  
    printf("\nSolution for the system:\n"); 
    for (int i=0; i<N; i++) 
        printf("%lf\n", x[i]); 
} 
  
// Driver program 
int main() 
{ 
    /* input matrix */
    double mat[N][N+1] = {{3.0, 2.0,-4.0, 3.0}, 
                          {2.0, 3.0, 3.0, 15.0}, 
                          {5.0, -3, 1.0, 14.0} 
                         }; 
  
    gaussianElimination(mat); 
  
    return 0; 
} 

6. Freivalid
#define MAX 512
#define MIN_TRIES 10

int a[MAX][MAX];
int b[MAX][MAX];
int c[MAX][MAX];
int N;

bool freivald() {
// Function to check if ABx = Cx
    // Generate a random vector
    bool r[N];
    for (int i = 0; i < N; i++)
        r[i] = rand() % 2;

    // Now comput B*r for evaluating
    // expression A * (B*r) - (C*r)
    int br[N];
    memset(br, 0, sizeof(int) * N);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            br[i] = br[i] + b[i][j] * r[j];

    // Now comput C*r for evaluating
    // expression A * (B*r) - (C*r)
    int cr[N];
    memset(cr, 0, sizeof(int) * N);

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            cr[i] = cr[i] + c[i][j] * r[j];

    // Now comput A* (B*r) for evaluating
    // expression A * (B*r) - (C*r)
    int axbr[N];
    memset(axbr, 0, sizeof(int) * N);

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            axbr[i] = axbr[i] + a[i][j] * br[j];

    // Finally check if value of expression
    // A * (B*r) - (C*r) is 0 or not
    for (int i = 0; i < N; i++)
        if (axbr[i] - cr[i] != 0)
            return false;

    return true;
}

7. sqrt Bigdecimal
static BigDecimal TWO = BigDecimal.ONE.add(BigDecimal.ONE);
public static BigDecimal sqrt(BigDecimal A, final int SCALE) {

    BigDecimal x0 = new BigDecimal("0");
    BigDecimal x1 = new BigDecimal(Math.sqrt(A.doubleValue()));
    while (!x0.equals(x1)) {
        x0 = x1;
        x1 = A.divide(x0, SCALE, BigDecimal.ROUND_HALF_UP);
        x1 = x1.add(x0);
        x1 = x1.divide(TWO, SCALE, BigDecimal.ROUND_HALF_UP);

    }
    return x1;
}

8. Mul/Power Matrix
vvi multiplyMatrix(vvi & arr, vvi & brr) {
    int n = arr.size();
    int m = brr[0].size();
    int p = brr.size();
    vvi res(n, vi(m, 0));
    FOR(i, 0, n) {
        FOR(j, 0, m) {
            FOR(k, 0, p) {
                res[i][j] = ((ll)res[i][j] + (ll)arr[i][k]*(ll)brr[k][j]) %  nMod;
            }
        }
    }
    return res;
}

vvi powerMatrix(vvi & arr, int pow) {
    if (pow == 0) {
        return vvi(arr.size(), vi(arr[0].size(), 0)); // Depend on case
    }
    if (pow == 1) {
        return arr;
    }
    vvi res = powerMatrix(arr, pow/2);
    res = multiplyMatrix(res, res);
    if (pow % 2 != 0) {
        res = multiplyMatrix(res, arr);
    }
    return res;
}

9. Fibonanci
// cal
vvi fibo(2, vi(1));
fibo[0][0] = 0;
fibo[1][0] = 1;
ll res = multiplyMatrix(pow, fibo)[0][0];

// Some identities:
// F(n+1)F(n-1) - F(n)^2 = -1^n
// F(n+k) = F(k)F(n+1) + F(k-1)F(n)
// F(2n-1) = F(n)^2 + F(n-1)^2
// SUM(i=0 to n)[F(i)] = F(n+2) - 1
// SUM(i=0 to n)[F(i)^2] = F(n)F(n+1)
// SUM(i=0 to n)[F(i)^3] = [F(n)F(n+1)^2 - (-1^n)F(n-1) + 1] / 2 
// gcd(Fm, Fn) = F(gcd(m,n))
// sqrt(5N^2 +- 4) is natural <-> exists natural k | F(k) = N
// [ F(0) F(1) ] [ [0 1] [1 1] ]^n = [ F(n) F(n+1) ]

bool miillerTest(int d, int n) 
{ 
    // Pick a random number in [2..n-2] 
    // Corner cases make sure that n > 4 
    int a = 2 + rand() % (n - 4); 
  
    // Compute a^d % n 
    int x = power(a, d, n); 
  
    if (x == 1  || x == n-1) 
       return true; 
  
    // Keep squaring x while one of the following doesn't 
    // happen 
    // (i)   d does not reach n-1 
    // (ii)  (x^2) % n is not 1 
    // (iii) (x^2) % n is not n-1 
    while (d != n-1) 
    { 
        x = (x * x) % n; 
        d *= 2; 
  
        if (x == 1)      return false; 
        if (x == n-1)    return true; 
    } 
  
    // Return composite 
    return false; 
}

//n is Fibinacci if one of 5*n*n + 4 or 5*n*n - 4 or both is a perferct square
 

10. Chinese Reainder Theorem

// Returns modulo inverse of a with respect to m using extended 
// Euclid Algorithm. Refer below post for details: 
// https://www.geeksforgeeks.org/multiplicative-inverse-under-modulo-m/ 
int inv(int a, int m) 
{ 
    int m0 = m, t, q; 
    int x0 = 0, x1 = 1; 
  
    if (m == 1) 
       return 0; 
  
    // Apply extended Euclid Algorithm 
    while (a > 1) 
    { 
        // q is quotient 
        q = a / m; 
  
        t = m; 
  
        // m is remainder now, process same as 
        // euclid's algo 
        m = a % m, a = t; 
  
        t = x0; 
  
        x0 = x1 - q * x0; 
  
        x1 = t; 
    } 
  
    // Make x1 positive 
    if (x1 < 0) 
       x1 += m0; 
  
    return x1; 
} 
  
// k is size of num[] and rem[].  Returns the smallest 
// number x such that: 
//  x % num[0] = rem[0], 
//  x % num[1] = rem[1], 
//  .................. 
//  x % num[k-2] = rem[k-1] 
// Assumption: Numbers in num[] are pairwise coprime 
// (gcd for every pair is 1) 
int findMinX(int num[], int rem[], int k) 
{ 
    // Compute product of all numbers 
    int prod = 1; 
    for (int i = 0; i < k; i++) 
        prod *= num[i]; 
  
    // Initialize result 
    int result = 0; 
  
    // Apply above formula 
    for (int i = 0; i < k; i++) 
    { 
        int pp = prod / num[i]; 
        result += rem[i] * inv(pp, num[i]) * pp; 
    } 
  
    return result % prod; 
} 
  
// Driver method 
int main(void) 
{ 
    int num[] = {3, 4, 5}; 
    int rem[] = {2, 3, 1}; 
    int k = sizeof(num)/sizeof(num[0]); 
    cout << "x is " << findMinX(num, rem, k); 
    return 0; 
} 

11. Combination

Catalan
The number of words of size 2n over the alphabet Σ = {a, b} having an equal number of a symbols and b symbols containing no prefix with more a symbols than b symbols.



2. Graph
1. Bridge & Articulation
#include <bits/stdc++.h>
using namespace std;

const int MAXN = 1e5 + 5;
int n;
vector<int> adj[MAXN];
int num[MAXN];
int low[MAXN];
int tms;
int root, nchild;

void dfs(int u, int p = -1) {
    num[u] = low[u] = ++tms;
    for (int i = 0; i < (int) adj[u].size(); i++) {
        int v = adj[u][i];
        if (!num[v]) {
            if (u == root) nchild++;
            dfs(v, u);
            if (low[v] >= num[u]) {
                //u is an articulation point
            }
            if (low[v] > num[u]) {
                //u -> v is a bridge
            }
            low[u] = min(low[u], low[v]);
        }
        else if (v != p) {
            low[u] = min(low[u], num[v]);
        }
    }
}

int main() {
    dfs(root = 0);
    if (nchild > 1) {
        //root is an articulation point
    }
    return 0;
}

2. Dijkstra + heap kc97ble

void dijkstra(){
    priority_queue <ii, vector<ii>, greater<ii> > pq;
    int i, u, v, du, uv;

    for (i=1; i<=n; i++) d[i] = oo;
    d[1] = 0;
    pq.push(ii(0, 1));

    while (pq.size()){
        u=pq.top().second;
        du=pq.top().first;
        pq.pop();
        if (du!=d[u]) continue;

        for (i=0; v=a[u][i].second; i++)
        {
            uv=a[u][i].first;
            if (d[v]>du+uv) {
                d[v]=du+uv;
                pq.push(ii(d[v], v));
            }
        }
    }

}

3. LCA kc97ble

const int N = 100005;
int n, Root, l[N], P[20][N];

int level(int u) {
    if (u==Root) return l[u]=1;
    if (l[u]==0) l[u]=level(P[0][u])+1;
    return l[u];
}

int lca(int x, int y) {
    for (int k=19; k>=0; k--) 
        if (l[P[k][x]]>=l[y]) 
        x=P[k][x];
    for (int k=19; k>=0; k--)
        if (l[P[k][y]]>=l[x]) 
        y=P[k][y];
    for (int k=19; k>=0; k--)
        if (P[k][x]!=P[k][y]) 
        { x=P[k][x]; y=P[k][y]; }
    while (x!=y)
        { x=P[0][x]; y=P[0][y]; }
    return x; 
}

void solve() {
    scanf("%d", &n);
    for (int i=1; i<=n; i++) {
        int p; scanf("%d", &p);
        while (p-->0) {
            int q; scanf("%d", &q);
            P[0][q] = i;
        }
    }
    for (int i=1; i<=n; i++)
        if (P[0][i]==0) Root=i;
    for (int i=1; i<=n; i++)
        level(i); // done l
    
    for (int k=1; k<=19; k++)
    for (int i=1; i<=n; i++)
    P[k][i] = P[k-1][P[k-1][i]];
    
    int m; scanf("%d", &m);
    while (m-->0) {
        int x, y;
        scanf("%d%d", &x, &y);
        printf("%d\n", lca(x, y));
    }
}

main() {
    int t; scanf("%d", &t);
    for (int i=1; i<=t; i++) {
        printf("Case %d:\n", i);
        solve();
        for (int j=1; j<=n; j++) 
        { l[j]=0; P[0][j]=0; }
    }
}

Mô tả
int l[]
l[u] là độ sâu của nút u, nút gốc có độ sâu bằng 1.
int P[][]
P[k][u] là tổ tiên thứ 2^k của u. Nói riêng, P[0][u] là cha trực tiếp của nút u.

4. Lehmer - Đếm số lượng số nguyên tố nhỏ hơn n

#define long long long
const int N = 100005;
const int M = 1000000007;
bool np[N];
int p[N], pp=0;

void eratos() {
    np[0]=np[1]=true;
    for (int i=2; i*i<N; i++) if (!np[i])
        for (int j=i*i; j<N; j+=i) np[j]=true;
    for (int i=2; i<N; i++)
    if (!np[i]) p[++pp]=i;
}

long power(long a, long k) {
    long P = 1;
    while (k) {
        if (k&1) P=P*a;
        k/=2; a=a*a;
    }
    return P;
}

long power(long a, long k, long M) {
    long P=1;
    for (a=a%M; k; k/=2)
    { if (k&1) P=P*a%M; a=a*a%M; }
    return P;
}

long root(long n, long k) {
    long x = pow(n, 1.0/k);
    while (power(x, k)%M==power(x, k, M) && power(x, k)<n) x++;
    while (power(x, k)%M!=power(x, k, M) || power(x, k)>n) x--;
    return x;
}

map<long, long> Phi[N];

long phi(long x, int a) {
    if (Phi[a].count(x)) return Phi[a][x];
    if (a==1) return (x+1)/2;
    long Result = phi(x, a-1) - phi(x/p[a], a-1);
    return Phi[a][x] = Result;
}

long pi(long x) {
    if (x<N)
        return upper_bound(p+1, p+pp+1, x) - (p+1);
    long a = pi(root(x, 4));
    long b = pi(root(x, 2));
    long c = pi(root(x, 3));
    long Sum = phi(x, a) + (b+a-2)*(b-a+1)/2;
    for (int i=a+1; i<=b; i++)
        Sum -= pi(x/p[i]);
    for (int i=a+1; i<=c; i++) {
        long bi = pi(root(x/p[i], 2));
        for (int j=i; j<=bi; j++)
        Sum -= pi(x/p[i]/p[j]) - (j-1);
    }
    return Sum;
}

main(){
    eratos();
    long n;
    while (cin >> n)
    cout << pi(n) << endl;
}

3. Geometry
1. Basic
#define double double
#define EPS 1e-9
struct PT {
    double x, y;
    PT() : x(0), y(0) {}
    PT(double x, double y) : x(x), y(y) {}
    PT(const PT& p) : x(p.x), y(p.y) {}
    int operator < (const PT& rhs) const {return make_pair(y, x) < make_pair(rhs.y, rhs.x);}
    int operator == (const PT& rhs) const {return make_pair(y, x) == make_pair(rhs.y, rhs.x);}
    PT operator + (const PT& p) const {return PT(x + p.x, y + p.y);}
    PT operator - (const PT& p) const {return PT(x - p.x, y - p.y);}
    PT operator * (double c) const {return PT(x * c, y * c);}
    PT operator / (double c) const {return PT(x / c, y / c);}
};
double cross(PT p, PT q) {return p.x * q.y - p.y * q.x;}
double area(PT a, PT b, PT c) {return fabs(cross(a, b) + cross(b, c) + cross(c, a)) / 2;}
double area2(PT a, PT b, PT c) {return cross(a, b) + cross(b, c) + cross(c, a);}
double dot(PT p, PT q) {return p.x * q.x + p.y * q.y;}
double dist(PT p, PT q) {return sqrt(dot(p - q, p - q));}
double dist2(PT p, PT q) {return dot(p - q, p - q);}
PT RotateCCW90(PT p) {return PT(-p.y, p.x);}
PT RotateCW90(PT p) {return PT(p.y, -p.x);}
PT RotateCCW(PT p, double t) {return PT(p.x * cos(t) - p.y * sin(t), p.x * sin(t) + p.y * cos(t));}
int sign(double x) {return x < -EPS ? -1 : x > EPS;}
int sign(double x, double y) {return sign(x - y);}
ostream& operator << (ostream& os, const PT& p) {
    os << "(" << p.x << "," << p.y << ")";
    return os;
}

//Project c on Line(a, b)
PT ProjectPointLine(PT a, PT b, PT c) {
    return a + (b - a) * dot(c - a, b - a) / dot(b - a, b - a);
}
PT ProjectPointSegment(PT a, PT b, PT c) {
    double r = dot(b - a, b - a);
    if (fabs(r) < EPS) return a;
    r = dot(c - a, b - a) / r;
    if (r < 0) return a;
    if (r > 1) return b;
    return a + (b - a) * r;
}
double DistancePointSegment(PT a, PT b, PT c) {
    return dist(c, ProjectPointSegment(a, b, c));
}
//Compute distance between PT (x, y, z) and plane ax + by + cz = d
double DistancePointPlane(double x, double y, double z, double a, double b, double c, double d) {
    return fabs(a * x + b * y + c * z - d) / sqrt(a * a + b * b + c * c);
}
//Determine if lines from a to b and c to d are parallel or collinear
int LinesParallel(PT a, PT b, PT c, PT d) {
    return fabs(cross(b - a, c - d)) < EPS;
}
int LinesCollinear(PT a, PT b, PT c, PT d) {
    return LinesParallel(a, b, c, d) && fabs(cross(a - b, a - c)) < EPS && fabs(cross(c - d, c - a)) < EPS;
}
//Determine if line segment from a to b intersects with line segment from c to d
int SegmentsIntersect(PT a, PT b, PT c, PT d) {
    if (LinesCollinear(a, b, c, d)) {
        if (dist2(a, c) < EPS || dist2(a, d) < EPS || dist2(b, c) < EPS || dist2(b, d) < EPS) return 1;
        if (dot(c - a, c - b) > 0 && dot(d - a, d - b) > 0 && dot(c - b, d - b) > 0) return 0;
        return 1;
    }
    if (cross(d - a, b - a) * cross(c - a, b - a) > 0) return 0;
    if (cross(a - c, d - c) * cross(b - c, d - c) > 0) return 0;
    return 1;
}
//Compute intersection of line passing through a and b
//with line passing through c and d, assuming that unique
//intersection exists; for segment intersection, check if
//segments intersect first
PT ComputeLineIntersection(PT a, PT b, PT c, PT d) {
    b = b - a; d = c - d; c = c - a;
    return a + b * cross(c, d) / cross(b, d);
}
//Compute center of circle given three points
PT ComputeCircleCenter(PT a, PT b, PT c) {
    b = (a + b) / 2;
    c = (a + c) / 2;
    return ComputeLineIntersection(b, b + RotateCW90(a - b), c, c + RotateCW90(a - c));
}
//Determine if point is in a possibly non-convex polygon
//returns 1 for strictly interior points, 0 for
//strictly exterior points, and 0 or 1 for the remaining points.
int PointInPolygonSlow(const vector<PT>& p, PT q) {
    int c = 0;
    for (int i = 0; i < p.size(); i++) {
        int j = (i + 1) % p.size();
        if ((p[i].y <= q.y && q.y < p[j].y || p[j].y <= q.y && q.y < p[i].y) && q.x < p[i].x + (p[j].x - p[i].x) * (q.y - p[i].y) / (p[j].y - p[i].y)) c = !c;
    }
    return c;
}
//Strictly inside convex Polygon
#define Det(a, b, c) ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x))
int PointInPolygon(vector<PT>& p, PT q) {
    int a = 1, b = p.size() - 1, c;
    if (Det(p[0], p[a], p[b]) > 0) swap(a, b);
    //Allow on edge --> if (Det... > 0 || Det ... < 0)
    if (Det(p[0], p[a], q) >= 0 || Det(p[0], p[b], q) <= 0) return 0;
    while(abs(a - b) > 1) {
        c = (a + b) / 2;
        if (Det(p[0], p[c], q) > 0) b = c; else a = c;
    }
    //Alow on edge --> return Det... <= 0
    return Det(p[a], p[b], q) < 0;
}
//Determine if point is on the boundary of a polygon
int PointOnPolygon(const vector<PT>& p, PT q) {
    for (int i = 0; i < p.size(); i++) if (dist2(ProjectPointSegment(p[i], p[(i + 1) % p.size()], q), q) < EPS) return 1;
    return 0;
}
//Compute intersection of line through points a and b with circle centered at c with radius r > 0
vector<PT> CircleLineIntersection(PT a, PT b, PT c, double r) {
    vector<PT> res;
    b = b - a; a = a - c;
    double A = dot(b, b);
    double B = dot(a, b);
    double C = dot(a, a) - r * r;
    double D = B * B - A * C;
    if (D < -EPS) return res;
    res.push_back(c + a + b * (-B + sqrt(D + EPS)) / A);
    if (D > EPS) res.push_back(c + a + b * (-B - sqrt(D)) / A);
    return res;
}
//Compute intersection of circle centered at a with radius r with circle centered at b with radius R
vector<PT> CircleCircleIntersection(PT a, PT b, double r, double R) {
    vector<PT> res;
    double d = sqrt(dist2(a, b));
    if (d > r + R || d + min(r, R) < max(r, R)) return res;
    double x = (d * d - R * R + r * r) / (2 * d);
    double y = sqrt(r * r - x * x);
    PT v = (b - a) / d;
    res.push_back(a + v * x + RotateCCW90(v) * y);
    if (y > 0) res.push_back(a + v * x - RotateCCW90(v) * y);
    return res;
}
//This code computes the area or centroid of a (possibly nonconvex)
//polygon, assuming that the coordinates are listed in a clockwise or
//counterclockwise fashion.  Note that the centroid is often known as
//the "center of gravity" or "center of mass".
double ComputeSignedArea(const vector<PT>& p) {
    double area = 0;
    for(int i = 0; i < p.size(); i++) {
        int j = (i + 1) % p.size();
        area += p[i].x * p[j].y - p[j].x * p[i].y;
    }
    return area / 2.0;
}
double ComputeArea(const vector<PT>& p) {
    return fabs(ComputeSignedArea(p));
}
PT ComputeCentroid(const vector<PT>& p) {
    PT c(0, 0);
    double scale = 6.0 * ComputeSignedArea(p);
    for (int i = 0; i < p.size(); i++) {
        int j = (i + 1) % p.size();
        c = c + (p[i] + p[j]) * (p[i].x * p[j].y - p[j].x * p[i].y);
    }
    return c / scale;
}
//Tests whether or not a given polygon (in CW or CCW order) is simple
int IsSimple(const vector<PT>& p) {
    for (int i = 0; i < p.size(); i++) {
        for (int k = i + 1; k < p.size(); k++) {
            int j = (i + 1) % p.size();
            int l = (k + 1) % p.size();
            if (i == l || j == k) continue;
            if (SegmentsIntersect(p[i], p[j], p[k], p[l])) return 0;
        }
    }
    return 1;
}
double Angle(PT a) {
    double PI = acos((double) - 1);
    if (a.x == 0) {
        if (a.y > 0) return PI / 2;
        return 3 * PI / 2;
    }
    if (a.y == 0) {
        if (a.x > 0) return 0;
        return PI;
    }
    double res = atan(a.y / a.x);
    if (a.x < 0) return res + PI;
    if (a.y < 0) return res + 2 * PI;
    return res;
}

int main() {
    // expected: (-5,2)
    cout << RotateCCW90(PT(2,5)) << endl;

    // expected: (5,-2)
    cout << RotateCW90(PT(2,5)) << endl;

    // expected: (-5,2)
    cout << RotateCCW(PT(2,5),M_PI/2) << endl;

    // expected: (5,2)
    cout << ProjectPointLine(PT(-5,-2), PT(10,4), PT(3,7)) << endl;

    // expected: (5,2) (7.5,3) (2.5,1)
    cout << ProjectPointSegment(PT(-5,-2), PT(10,4), PT(3,7)) << " "
         << ProjectPointSegment(PT(7.5,3), PT(10,4), PT(3,7)) << " "
         << ProjectPointSegment(PT(-5,-2), PT(2.5,1), PT(3,7)) << endl;

    // expected: 6.78903
    cout << DistancePointPlane(4,-4,3,2,-2,5,-8) << endl;

    // expected: 1 0 1
    cout << LinesParallel(PT(1,1), PT(3,5), PT(2,1), PT(4,5)) << " "
         << LinesParallel(PT(1,1), PT(3,5), PT(2,0), PT(4,5)) << " "
         << LinesParallel(PT(1,1), PT(3,5), PT(5,9), PT(7,13)) << endl;

    // expected: 0 0 1
    cout << LinesCollinear(PT(1,1), PT(3,5), PT(2,1), PT(4,5)) << " "
         << LinesCollinear(PT(1,1), PT(3,5), PT(2,0), PT(4,5)) << " "
         << LinesCollinear(PT(1,1), PT(3,5), PT(5,9), PT(7,13)) << endl;

    // expected: 1 1 1 0
    cout << SegmentsIntersect(PT(0,0), PT(2,4), PT(3,1), PT(-1,3)) << " "
         << SegmentsIntersect(PT(0,0), PT(2,4), PT(4,3), PT(0,5)) << " "
         << SegmentsIntersect(PT(0,0), PT(2,4), PT(2,-1), PT(-2,1)) << " "
         << SegmentsIntersect(PT(0,0), PT(2,4), PT(5,5), PT(1,7)) << endl;

    // expected: (1,2)
    cout << ComputeLineIntersection(PT(0,0), PT(2,4), PT(3,1), PT(-1,3)) << endl;

    // expected: (1,1)
    cout << ComputeCircleCenter(PT(-3,4), PT(6,1), PT(4,5)) << endl;

    vector<PT> v;
    v.push_back(PT(0,0));
    v.push_back(PT(5,0));
    v.push_back(PT(5,5));
    v.push_back(PT(0,5));

    // expected: 1 1 1 0 0
    cout << PointInPolygonSlow(v, PT(2,2)) << " "
         << PointInPolygonSlow(v, PT(2,0)) << " "
         << PointInPolygonSlow(v, PT(0,2)) << " "
         << PointInPolygonSlow(v, PT(5,2)) << " "
         << PointInPolygonSlow(v, PT(2,5)) << endl;

    // expected: 0 1 1 1 1
    cout << PointOnPolygon(v, PT(2,2)) << " "
         << PointOnPolygon(v, PT(2,0)) << " "
         << PointOnPolygon(v, PT(0,2)) << " "
         << PointOnPolygon(v, PT(5,2)) << " "
         << PointOnPolygon(v, PT(2,5)) << endl;

    // expected: (1,6)
    //           (5,4) (4,5)
    //           blank line
    //           (4,5) (5,4)
    //           blank line
    //           (4,5) (5,4)
    vector<PT> u = CircleLineIntersection(PT(0,6), PT(2,6), PT(1,1), 5);
    for (int i = 0; i < u.size(); i++) cerr << u[i] << " ";
    cout << endl;
    u = CircleLineIntersection(PT(0,9), PT(9,0), PT(1,1), 5);
    for (int i = 0; i < u.size(); i++) cerr << u[i] << " ";
    cout << endl;
    u = CircleCircleIntersection(PT(1,1), PT(10,10), 5, 5);
    for (int i = 0; i < u.size(); i++) cerr << u[i] << " ";
    cout << endl;
    u = CircleCircleIntersection(PT(1,1), PT(8,8), 5, 5);
    for (int i = 0; i < u.size(); i++) cerr << u[i] << " ";
    cout << endl;
    u = CircleCircleIntersection(PT(1,1), PT(4.5,4.5), 10, sqrt(2.0)/2.0);
    for (int i = 0; i < u.size(); i++) cerr << u[i] << " ";
    cout << endl;
    u = CircleCircleIntersection(PT(1,1), PT(4.5,4.5), 5, sqrt(2.0)/2.0);
    for (int i = 0; i < u.size(); i++) cerr << u[i] << " ";
    cout << endl;

    // area should be 5.0
    // centroid should be (1.1666666, 1.166666)
    PT pa[] = { PT(0,0), PT(5,0), PT(1,1), PT(0,5) };
    vector<PT> p(pa, pa+4);
    PT c = ComputeCentroid(p);
    cout << "Area: " << ComputeArea(p) << endl;
    cout << "Centroid: " << c << endl;
    return 0;
}

2. ConvexHull
//Remove degenerate
#define REMOVE_REDUNDANT
#ifdef REMOVE_REDUNDANT
bool between(const PT& a, const PT& b, const PT& c) {
    return (fabs(area2(a, b, c)) < EPS && (a.x - b.x) * (c.x - b.x) <= 0 && (a.y - b.y) * (c.y - b.y) <= 0);
}
#endif
void ConvexHull(vector<PT>& pts) {
    sort(pts.begin(), pts.end());
    pts.erase(unique(pts.begin(), pts.end()), pts.end());
    vector<PT> up, dn;
    for (int i = 0; i < pts.size(); i++) {
        while (up.size() > 1 && area2(up[up.size() - 2], up.back(), pts[i]) >= 0) up.pop_back();
        while (dn.size() > 1 && area2(dn[dn.size() - 2], dn.back(), pts[i]) <= 0) dn.pop_back();
        up.push_back(pts[i]);
        dn.push_back(pts[i]);
    }
    pts = dn;
    for (int i = up.size() - 2; i >= 1; i--) pts.push_back(up[i]);
#ifdef REMOVE_REDUNDANT
    if (pts.size() <= 2) return;
    dn.clear();
    dn.push_back(pts[0]);
    dn.push_back(pts[1]);
    for (int i = 2; i < pts.size(); i++) {
        if (between(dn[dn.size() - 2], dn[dn.size() - 1], pts[i])) dn.pop_back();
        dn.push_back(pts[i]);
    }
    if (dn.size() >= 3 && between(dn.back(), dn[0], dn[1])) {
        dn[0] = dn.back();
        dn.pop_back();
    }
    pts = dn;
#endif
}

4. Data Structure
1. BIT 2D
2D BIT is basically a BIT where each element is another BIT. 
Updating by adding v on (x, y) means it's effect will be found 
throughout the rectangle [(x, y), (max_x, max_y)], 
and query for (x, y) gives you the result of the rectangle 
[(0, 0), (x, y)], assuming the total rectangle is 
[(0, 0), (max_x, max_y)]. So when you query and update on 
this BIT,you have to be careful about how many times you are 
subtracting a rectangle and adding it. Simple set union formula 
works here. 
  
So if you want to get the result of a specific rectangle 
[(x1, y1), (x2, y2)], the following steps are necessary: 
  
Query(x1,y1,x2,y2) = getSum(x2, y2)-getSum(x2, y1-1) - 
                     getSum(x1-1, y2)+getSum(x1-1, y1-1) 
  
Here 'Query(x1,y1,x2,y2)' means the sum of elements enclosed 
in the rectangle with bottom-left corner's co-ordinates 
(x1, y1) and top-right corner's co-ordinates - (x2, y2) 
  
Constraints -> x1<=x2 and y1<=y2 
  
    /\ 
 y  | 
    |           --------(x2,y2) 
    |          |       | 
    |          |       | 
    |          |       | 
    |          --------- 
    |       (x1,y1) 
    | 
    |___________________________ 
   (0, 0)                   x--> 
  
In this progrm we have assumed a square matrix. The 
program can be easily extended to a rectangular one. */
  
#include<bits/stdc++.h> 
using namespace std; 
  
#define N 4 // N-->max_x and max_y 
  
// A structure to hold the queries 
struct Query 
{ 
    int x1, y1; // x and y co-ordinates of bottom left 
    int x2, y2; // x and y co-ordinates of top right 
}; 
  
// A function to update the 2D BIT 
void updateBIT(int BIT[][N+1], int x, int y, int val) 
{ 
    for (; x <= N; x += (x & -x)) 
    { 
        // This loop update all the 1D BIT inside the 
        // array of 1D BIT = BIT[x] 
        for (; y <= N; y += (y & -y)) 
            BIT[x][y] += val; 
    } 
    return; 
} 
  
// A function to get sum from (0, 0) to (x, y) 
int getSum(int BIT[][N+1], int x, int y) 
{ 
    int sum = 0; 
  
    for(; x > 0; x -= x&-x) 
    { 
        // This loop sum through all the 1D BIT 
        // inside the array of 1D BIT = BIT[x] 
        for(; y > 0; y -= y&-y) 
        { 
            sum += BIT[x][y]; 
        } 
    } 
    return sum; 
} 
  
// A function to create an auxiliary matrix 
// from the given input matrix 
void constructAux(int mat[][N], int aux[][N+1]) 
{ 
    // Initialise Auxiliary array to 0 
    for (int i=0; i<=N; i++) 
        for (int j=0; j<=N; j++) 
            aux[i][j] = 0; 
  
    // Construct the Auxiliary Matrix 
    for (int j=1; j<=N; j++) 
        for (int i=1; i<=N; i++) 
            aux[i][j] = mat[N-j][i-1]; 
  
    return; 
} 
  
// A function to construct a 2D BIT 
void construct2DBIT(int mat[][N], int BIT[][N+1]) 
{ 
    // Create an auxiliary matrix 
    int aux[N+1][N+1]; 
    constructAux(mat, aux); 
  
    // Initialise the BIT to 0 
    for (int i=1; i<=N; i++) 
        for (int j=1; j<=N; j++) 
            BIT[i][j] = 0; 
  
    for (int j=1; j<=N; j++) 
    { 
        for (int i=1; i<=N; i++) 
        { 
            // Creating a 2D-BIT using update function 
            // everytime we/ encounter a value in the 
            // input 2D-array 
            int v1 = getSum(BIT, i, j); 
            int v2 = getSum(BIT, i, j-1); 
            int v3 = getSum(BIT, i-1, j-1); 
            int v4 = getSum(BIT, i-1, j); 
  
            // Assigning a value to a particular element 
            // of 2D BIT 
            updateBIT(BIT, i, j, aux[i][j]-(v1-v2-v4+v3)); 
        } 
    } 
  
    return; 
} 
  
// A function to answer the queries 
void answerQueries(Query q[], int m, int BIT[][N+1]) 
{ 
    for (int i=0; i<m; i++) 
    { 
        int x1 = q[i].x1 + 1; 
        int y1 = q[i].y1 + 1; 
        int x2 = q[i].x2 + 1; 
        int y2 = q[i].y2 + 1; 
  
        int ans = getSum(BIT, x2, y2)-getSum(BIT, x2, y1-1)- 
                  getSum(BIT, x1-1, y2)+getSum(BIT, x1-1, y1-1); 
  
        printf ("Query(%d, %d, %d, %d) = %d\n", 
                q[i].x1, q[i].y1, q[i].x2, q[i].y2, ans); 
    } 
    return; 
} 
  
// Driver program 
int main() 
{ 
    int mat[N][N] = {{1, 2, 3, 4}, 
                    {5, 3, 8, 1}, 
                    {4, 6, 7, 5}, 
                    {2, 4, 8, 9}}; 
  
    // Create a 2D Binary Indexed Tree 
    int BIT[N+1][N+1]; 
    construct2DBIT(mat, BIT); 
  
    Query q[] = {{1, 1, 3, 2}, {2, 3, 3, 3}, {1, 1, 1, 1}}; 
    int m = sizeof(q)/sizeof(q[0]); 
  
    answerQueries(q, m, BIT); 
  
    return(0); 
}
2. Persistent Tree
#define MAXN 100 
  
/* data type for individual 
 * node in the segment tree */
struct node 
{ 
    // stores sum of the elements in node 
    int val; 
  
    // pointer to left and right children 
    node* left, *right; 
  
    // required constructors........ 
    node() {} 
    node(node* l, node* r, int v) 
    { 
        left = l; 
        right = r; 
        val = v; 
    } 
}; 
  
// input array 
int arr[MAXN]; 
  
// root pointers for all versions 
node* version[MAXN]; 
  
// Constructs Version-0 
// Time Complexity : O(nlogn) 
void build(node* n,int low,int high) 
{ 
    if (low==high) 
    { 
        n->val = arr[low]; 
        return; 
    } 
    int mid = (low+high) / 2; 
    n->left = new node(NULL, NULL, 0); 
    n->right = new node(NULL, NULL, 0); 
    build(n->left, low, mid); 
    build(n->right, mid+1, high); 
    n->val = n->left->val + n->right->val; 
} 
  
/** 
 * Upgrades to new Version 
 * @param prev : points to node of previous version 
 * @param cur  : points to node of current version 
 * Time Complexity : O(logn) 
 * Space Complexity : O(logn)  */
void upgrade(node* prev, node* cur, int low, int high, 
                                   int idx, int value) 
{ 
    if (idx > high or idx < low or low > high) 
        return; 
  
    if (low == high) 
    { 
        // modification in new version 
        cur->val = value; 
        return; 
    } 
    int mid = (low+high) / 2; 
    if (idx <= mid) 
    { 
        // link to right child of previous version 
        cur->right = prev->right; 
  
        // create new node in current version 
        cur->left = new node(NULL, NULL, 0); 
  
        upgrade(prev->left,cur->left, low, mid, idx, value); 
    } 
    else
    { 
        // link to left child of previous version 
        cur->left = prev->left; 
  
        // create new node for current version 
        cur->right = new node(NULL, NULL, 0); 
  
        upgrade(prev->right, cur->right, mid+1, high, idx, value); 
    } 
  
    // calculating data for current version 
    // by combining previous version and current 
    // modification 
    cur->val = cur->left->val + cur->right->val; 
} 
  
int query(node* n, int low, int high, int l, int r) 
{ 
    if (l > high or r < low or low > high) 
       return 0; 
    if (l <= low and high <= r) 
       return n->val; 
    int mid = (low+high) / 2; 
    int p1 = query(n->left,low,mid,l,r); 
    int p2 = query(n->right,mid+1,high,l,r); 
    return p1+p2; 
} 
  
int main(int argc, char const *argv[]) 
{ 
    int A[] = {1,2,3,4,5}; 
    int n = sizeof(A)/sizeof(int); 
  
    for (int i=0; i<n; i++)  
       arr[i] = A[i]; 
  
    // creating Version-0 
    node* root = new node(NULL, NULL, 0); 
    build(root, 0, n-1); 
  
    // storing root node for version-0 
    version[0] = root; 
  
    // upgrading to version-1 
    version[1] = new node(NULL, NULL, 0); 
    upgrade(version[0], version[1], 0, n-1, 4, 1); 
  
    // upgrading to version-2 
    version[2] = new node(NULL, NULL, 0); 
    upgrade(version[1],version[2], 0, n-1, 2, 10); 
  
    cout << "In version 1 , query(0,4) : "; 
    cout << query(version[1], 0, n-1, 0, 4) << endl; 
  
    cout << "In version 2 , query(3,4) : "; 
    cout << query(version[2], 0, n-1, 3, 4) << endl; 
  
    cout << "In version 0 , query(0,3) : "; 
    cout << query(version[0], 0, n-1, 0, 3) << endl; 
    return 0; 
} 


5. String

1. KMP
#define MAX_N 100010

char T[MAX_N], P[MAX_N]; // T = text, P = pattern
int b[MAX_N], n, m; // b = back table, n = length of T, m = length of P

void kmpPreprocess() { // call this before calling kmpSearch()
  int i = 0, j = -1; b[0] = -1; // starting values
  while (i < m) { // pre-process the pattern string P
    while (j >= 0 && P[i] != P[j]) j = b[j]; // if different, reset j using b
    i++; j++; // if same, advance both pointers
    b[i] = j; // observe i = 8, 9, 10, 11, 12 with j = 0, 1, 2, 3, 4
} }           // in the example of P = "SEVENTY SEVEN" above

void kmpSearch() { // this is similar as kmpPreprocess(), but on string T
  int i = 0, j = 0; // starting values
  while (i < n) { // search through string T
    while (j >= 0 && T[i] != P[j]) j = b[j]; // if different, reset j using b
    i++; j++; // if same, advance both pointers
    if (j == m) { // a match found when j == m
      printf("P is found at index %d in T\n", i - j);
      j = b[j]; // prepare j for the next possible match
} } }

int main() {
  strcpy(T, "I DO NOT LIKE SEVENTY SEV BUT SEVENTY SEVENTY SEVEN");
  strcpy(P, "SEVENTY SEVEN");
  n = (int)strlen(T);
  m = (int)strlen(P);

  printf("T = '%s'\n", T);
  printf("P = '%s'\n", P);
  printf("\n");
  kmpPreprocess();
  kmpSearch();
  return 0;
}


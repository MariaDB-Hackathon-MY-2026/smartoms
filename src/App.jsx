import React, { useState, useEffect, useRef } from 'react';
import './App.css';

// ============================================================
// 📡 API CONFIGURATION
// ============================================================
const API_CONFIG = {
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000/api',
  endpoints: {
    checkUserStatus: '/users/check-status',
    getUserProfile: '/users/profile',
    products: '/products',
    recommendations: '/recommendations',
    popular: '/popular',
    views: '/views',
    orders: '/orders',
  },
  NEW_USER_THRESHOLD_HOURS: 24,
};

// ============================================================
// 🛍️ APP COMPONENT
// ============================================================

function App() {
  // ================= STATE MANAGEMENT =================
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [currentUser, setCurrentUser] = useState(null);
  const [cart, setCart] = useState([]);
  
  // UI State
  const [isAuthModalOpen, setIsAuthModalOpen] = useState(false);
  const [isCartOpen, setIsCartOpen] = useState(false);
  const [showColdStart, setShowColdStart] = useState(false);
  const [showToast, setShowToast] = useState(false);
  const [toastMsg, setToastMsg] = useState('');
  const [toastType, setToastType] = useState('success');
  
  // Form State
  const [authMode, setAuthMode] = useState('login'); // 'login' or 'signup'
  const [loginEmail, setLoginEmail] = useState('');
  const [loginPass, setLoginPass] = useState('');
  const [signupName, setSignupName] = useState('');
  const [signupEmail, setSignupEmail] = useState('');
  const [signupPass, setSignupPass] = useState('');
  
  // Dynamic State
  const [recommendationsTitle, setRecommendationsTitle] = useState("Featured Products");
  const [recommendationsSubtitle, setRecommendationsSubtitle] = useState("Handpicked selections just for you");
  const [displayedProducts, setDisplayedProducts] = useState([]);
  const [wishlist, setWishlist] = useState([]);

  // ================= EFFECTS =================
  
  const isFirstRender = useRef(true);

  // Single mount effect
  useEffect(() => {
    checkExistingSession();

    const savedWishlist = localStorage.getItem('smartoms_wishlist');
    if (savedWishlist) {
      setWishlist(JSON.parse(savedWishlist));
    }

    const savedCart = localStorage.getItem('smartoms_cart');
    if (savedCart) {
      setCart(JSON.parse(savedCart));
    }
  }, []);

  // Save cart (skip first render to avoid overwriting on mount)
  useEffect(() => {
    if (isFirstRender.current) {
      isFirstRender.current = false;
      return;
    }
    localStorage.setItem('smartoms_cart', JSON.stringify(cart));
  }, [cart]);

  // Fetch products/recommendations
  useEffect(() => {
    if (currentUser) {
      fetchRecommendations();
    } else {
      fetchAndRenderProducts();
    }
  }, [selectedCategory, currentUser]);

  // ================= API CALLS =================

  const fetchAndRenderProducts = async () => {
    try {
      let url = '';

      if (selectedCategory === 'all') {
        // FETCH ALL 500 ITEMS
        url = `${API_CONFIG.baseURL}/products?limit=500`;
        setRecommendationsTitle("All Products");
        setRecommendationsSubtitle("Full Catalog");
      } else {
        // Categories can keep limit 20 for performance, or increase if needed
        url = `${API_CONFIG.baseURL}/products?category=${selectedCategory}&limit=133`;
        setRecommendationsTitle(`${selectedCategory}`);
      }

      setRecommendationsSubtitle(
        selectedCategory === 'all' 
          ? "Full Catalog" 
          : `Showing items in ${selectedCategory}`
      );
        
      const response = await fetch(url);
      const data = await response.json();
      
      const items = data.products || [];
      setDisplayedProducts(items);
      
    } catch (error) {
      console.error("Failed to load products:", error);
      triggerToast("Failed to load products", "error");
    }
  };

  const handleCategoryChange = (category) => {
    if (category !== 'all' && !currentUser) {
      setIsAuthModalOpen(true);
      triggerToast("Please login to explore categories", "info");
      return;
    }
    setSelectedCategory(category);
  };

  const fetchRecommendations = async () => {
    if (!currentUser) return;

    const isNewUser = checkIfNewUser(currentUser);
    setShowColdStart(isNewUser);

    if (selectedCategory === 'all') {
      fetchAndRenderProducts();
      return;
    }

    // Category tabs — just fetch all products in that category
    if (selectedCategory !== 'recs') {
      const response = await fetch(
        `${API_CONFIG.baseURL}/products?category=${selectedCategory}&limit=500`
      );
      const data = await response.json();
      setDisplayedProducts(data.products || []);
      setRecommendationsTitle(selectedCategory.charAt(0).toUpperCase() + selectedCategory.slice(1));
      setRecommendationsSubtitle(`All products in ${selectedCategory}`);
      return;
    }

    // Recs tab only
    try {
      let recs = [];

      if (isNewUser) {
        const response = await fetch(
          `${API_CONFIG.baseURL}/popular?limit=20`
        );
        const data = await response.json();
        recs = data.popular || [];
        setRecommendationsTitle("Most Popular");
        setRecommendationsSubtitle("All-time best sellers");
        setShowColdStart(true);
      } else {
        const response = await fetch(
          `${API_CONFIG.baseURL}/recommendations/${currentUser.user_id}?n=20`
        );
        const data = await response.json();
        recs = data.recommendations || [];
        setRecommendationsTitle("Recommended For You");
        setRecommendationsSubtitle(`Based on your history (${data.algorithm})`);
        setShowColdStart(false);
      }

      setDisplayedProducts(recs);

    } catch (error) {
      console.error("Error fetching recs:", error);
    }
  };

  // ================= AUTH LOGIC =================

  const checkExistingSession = () => {
    const savedUser = localStorage.getItem('smartoms_user');
    if (savedUser) {
      setCurrentUser(JSON.parse(savedUser));
    }
  };

  const checkIfNewUser = (user) => {
    if (!user.createdAt) return true;
    const created = new Date(user.createdAt);
    const now = new Date();
    const hours = (now - created) / (1000 * 60 * 60);
    return hours < API_CONFIG.NEW_USER_THRESHOLD_HOURS;
  };

  const handleUserLoginFlow = async () => {
    // Trigger the recommendation fetch logic inside useEffect
    // We split this out to keep useEffect clean
    // Note: In production, you might want to verify token with backend here first
  };

  const handleLogin = async (e) => {
    e.preventDefault();

    try {
      const response = await fetch(`${API_CONFIG.baseURL}/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email: loginEmail,
          password: loginPass
        })
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Login failed");
      }

      const realUser = {
        user_id: data.user_id,     // Real ID from DB
        name: data.name,            // Real name from DB
        email: data.email,          // Real email
        createdAt: data.created_at  // Real Join Date (For Cold Start logic)
      };

      setCurrentUser(realUser);
      localStorage.setItem('smartoms_user', JSON.stringify(realUser));
      setIsAuthModalOpen(false);
      triggerToast(`Welcome back, ${realUser.name}!`, 'success');

    } catch (error) {
      console.error("Login error:", error);
      triggerToast(error.message || "Failed to login", "error");
    }
  };

  const handleSignup = async (e) => {
      e.preventDefault();

      const name = signupName;
      const email = signupEmail;
      const password = signupPass;

      try {

          const response = await fetch(`${API_CONFIG.baseURL}/users`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                  name: name,
                  email: email,
                  password: password
              })
          });

          const data = await response.json();

          if (!response.ok) {
              throw new Error(data.detail || "Signup failed");
          }

          const newUser = {
              user_id: data.user_id,
              name: data.name,
              email: data.email,
              createdAt: new Date().toISOString() // This matches the DB 'created_at'
          };

          setCurrentUser(newUser);
          localStorage.setItem('smartoms_user', JSON.stringify(newUser));

          setIsAuthModalOpen(false);
          triggerToast(`Account created! Welcome, ${name}! 🎉`, 'success');

      } catch (error) {
          console.error("Signup error:", error);
          triggerToast(error.message || "Failed to create account", "error");
      }
  };

  const logout = () => {
    setCurrentUser(null);
    setCart([]);
    localStorage.removeItem('smartoms_user');
    localStorage.removeItem('smartoms_cart'); // Add this
    setShowColdStart(false);
    triggerToast('Logged out successfully', 'info');
    setSelectedCategory('all');
    fetchAndRenderProducts();
  };

  // ================= CART LOGIC =================

  const addToCart = (name, price, id) => {

    if (!currentUser) {
      setIsAuthModalOpen(true); // Opens the popup
      triggerToast("Please login to add items to cart", "info");
      return;
    }

    setCart(prevCart => {
      const existing = prevCart.find(item => item.name === name);
      if (existing) {
        return prevCart.map(item => 
          item.name === name ? { ...item, quantity: item.quantity + 1 } : item
        );
      }
      return [...prevCart, { name, price, quantity: 1, id }];
    });
    triggerToast(`${name} added to cart`, 'success');
    
    // Send to API for training
    if (currentUser && id) {
        createOrder(id, "General"); // Placeholder category
    }
  };

  const toggleWishlist = (pid) => {
    // 1. Check Auth
    if (!currentUser) {
      setIsAuthModalOpen(true);
      triggerToast("Please login to save wishlist", "info");
      return;
    }

    // 2. Toggle Logic
    setWishlist(prevWishlist => {
      const exists = prevWishlist.includes(pid);
      let newWishlist = [];

      if (exists) {
        // Remove if already there
        newWishlist = prevWishlist.filter(id => id !== pid);
        triggerToast('Removed from wishlist', 'warning');
      } else {
        // Add if not there
        newWishlist = [...prevWishlist, pid];
        triggerToast('Added to wishlist', 'success');
      }

      // 3. Save to LocalStorage
      localStorage.setItem('smartoms_wishlist', JSON.stringify(newWishlist));
      return newWishlist;
    });
  };

  const removeFromCart = (name) => {
    setCart(prevCart => prevCart.filter(item => item.name !== name));
  };

  const createOrder = async (productId, category) => {
    try {
        await fetch(`${API_CONFIG.baseURL}/orders`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_id: currentUser.user_id,
                product_id: productId,
                quantity: 1,
                rating_score: 5.0
            })
        });
    } catch(e) { console.error(e); }
  };

  // ================= HELPERS =================

  const triggerToast = (msg, type = 'success') => {
    setToastMsg(msg);
    setToastType(type);
    setShowToast(true);
    setTimeout(() => setShowToast(false), 3000);
  };

  // ================= RENDER HELPERS =================

  const renderProductCard = (product, isRecommendation = false) => {
    // Destructure the data from the backend
    const pid = product.product_id;
    const name = product.product_name || `Product ${pid}`;
    const cat = product.category || 'General';
    const price = product.price ?? product.catalog_price ?? 0;
    const desc = product.product_description || product.description || "No description available.";
    const quantity = product.quantity_available ?? product.stock ?? product.quantity ?? 0;

    const isTrendingItem = !!product.trending_score;

    return (
      <div key={pid} className="product-card" data-category={cat}>
        <div className="product-image">
          <i className="fas fa-box"></i>

          {isTrendingItem && <span className="product-badge" style={{background:'linear-gradient(45deg, #ff0055, #ff00aa)'}}>Trending</span>}
          {isRecommendation && <span className="product-badge" style={{background:'#007bff'}}>Top Pick!</span>}
            <div 
              className="product-wishlist" 
              onClick={(e) => { 
                e.stopPropagation(); 
                if (!currentUser) {
                    setIsAuthModalOpen(true); // Opens popup
                    return;
                }

                toggleWishlist(pid);
              }}
            >
            <i 
              className={wishlist.includes(pid) ? "fas fa-heart" : "far fa-heart"}
              style={{ color: wishlist.includes(pid) ? "#ff0000" : "inherit" }}
            ></i>
          </div>
        </div>
        
        <div className="product-info">
          <div className="product-category">{cat}</div>
          <h3 className="product-name">{name}</h3>
          
          <p className="product-description" style={{fontSize:'0.85rem', color:'#666', margin:'0.5rem 0', minHeight:'40px', display:'-webkit-box', WebkitLineClamp:2, WebkitBoxOrient:'vertical', overflow:'hidden'}}>
            {desc.length > 60 ? desc.substring(0, 60) + "..." : desc}
          </p>

          <div className="product-rating">
            <i className="fas fa-star"></i>
            <span className="rating-count">
              {Math.round(product.weighted_rating || product.avg_rating || 0)} 
              ({product.num_ratings || 0})
            </span>
          </div>

          <div className="product-price">
            <span className="current-price">RM{parseFloat(price).toFixed(2)}</span>
          </div>

          {/* NEW: Handle Stock Logic */}
          <button 
            className="add-to-cart-btn" 
            onClick={() => addToCart(name, price, pid)}
            disabled={quantity <= 0} // Disable button if out of stock
            style={quantity <= 0 ? { backgroundColor: '#ccc', cursor: 'not-allowed' } : {}}
          >
            <i className="fas fa-shopping-cart"></i> 
            {quantity > 0 ? "Add to Cart" : "Out of Stock"}
          </button>
        </div>
      </div>
    );
  };

  // ================= MAIN JSX RETURN =================

  return (
    <div className="app-container">
      
      {/* Header */}
      <header className="header">
        <div className="header-container">
          <div className="logo">SMART<span>OMS</span></div>

          {/* Nav Icons */}
          <div className="nav-icons">
            
            {/* 1. Profile Icon + Name */}
            <div 
              className="profile-icon" 
              onClick={() => !currentUser ? setIsAuthModalOpen(true) : triggerToast('Profile Menu', 'info')}
              style={{ marginRight: '1.5rem' }} // Spacing
            >
              <i className="fas fa-user"></i>
              
              {/* Name Text - Simplified to prevent layout break */}
              {currentUser && (
                <span style={{ marginLeft: '10px', fontSize: '0.9rem', color: '#fff', verticalAlign: 'middle' }}>
                  {currentUser.name}
                </span>
              )}
            </div>

            {/* 2. Cart Icon */}
            <div 
              className="cart-icon" 
              onClick={() => setIsCartOpen(true)}
              style={{ marginRight: '1.5rem' }} // Spacing
            >
              <i className="fas fa-shopping-cart"></i>
              <span className="cart-count">{cart.reduce((a, b) => a + b.quantity, 0)}</span>
            </div>

            {/* 3. Logout Icon */}
            {currentUser && (
              <div 
                className="profile-icon" // Reuse profile-icon styles
                onClick={logout}
              >
                <i className="fas fa-sign-out-alt"></i>
              </div>
            )}

          </div>
        </div>
      </header>

      {/* Cold Start Banner */}
      {showColdStart && (
        <div className="cold-start-banner show">
          <div className="cold-start-container">
            <div className="cold-start-content">
              <div className="cold-start-icon"><i className="fas fa-snowflake"></i></div>
              <div className="cold-start-text">
                <h3>Welcome to SmartOMS!</h3>
                <p>You're new here! Let's get you started.</p>
              </div>
            </div>
            <button className="cold-start-btn" onClick={() => setShowColdStart(false)}>Dismiss</button>
          </div>
        </div>
      )}

      {/* Category Tabs */}
      <div className="category-tabs">
        <div className="tabs-container">

          <button 
            className={`tab-btn ${selectedCategory === 'recs' ? 'active' : ''} ${!currentUser ? 'locked' : ''}`}
            onClick={() => handleCategoryChange('recs')}
          >
            Recommended
          </button>

          <button 
            className={`tab-btn ${selectedCategory === 'all' ? 'active' : ''}`} 
            onClick={() => handleCategoryChange('all')} // 'all' stays free
          >
            All Products
          </button>

          <button 
            className={`tab-btn ${selectedCategory === 'electronics' ? 'active' : ''} ${!currentUser ? 'locked' : ''}`}
            onClick={() => handleCategoryChange('electronics')}
          >
            Electronics
          </button>

          <button 
            className={`tab-btn ${selectedCategory === 'books' ? 'active' : ''} ${!currentUser ? 'locked' : ''}`}
            onClick={() => handleCategoryChange('books')}
          >
            Books
          </button>

          <button 
            className={`tab-btn ${selectedCategory === 'clothing' ? 'active' : ''} ${!currentUser ? 'locked' : ''}`} 
            onClick={() => handleCategoryChange('clothing')}
          >
            Clothing
          </button>

        </div>
      </div>

      {/* Hero Section */}
      <section className="hero">
        <div className="hero-content">
          <h1>Welcome to SMARTOMS</h1>
          <p>Discover premium products at unbeatable prices.</p>
        </div>
      </section>

      {/* Products Section */}
      <section className="products-section" id="products">
        <div className="section-header">
          <h2>{recommendationsTitle}</h2>
          <p>{recommendationsSubtitle}</p>
        </div>

        <div className="product-grid">
          {displayedProducts.map(p => renderProductCard(p, recommendationsTitle.includes("Recommended")))}
        </div>

        {selectedCategory === 'all' && displayedProducts.length === 500 && (
          <div style={{textAlign:'center', padding:'2rem', color:'#666'}}>
            End of catalog (500 items)
          </div>
        )}
      </section>

      {/* Footer */}
      <footer className="footer">
        <div className="footer-container">
           <div className="footer-section">
             <h4>About SMARTOMS</h4>
             <p style={{color:'#ccc'}}>Your premier destination for quality products.</p>
           </div>
        </div>
        <div className="footer-bottom">
          <p>&copy; 2026 SMARTOMS. All rights reserved.</p>
        </div>
      </footer>

      {/* Auth Modal */}
      {isAuthModalOpen && (
        <div className="auth-overlay show" onClick={() => setIsAuthModalOpen(false)}>
          <div className="auth-modal" onClick={e => e.stopPropagation()}>
            <div className="auth-modal-header">
              <button className="auth-close-btn" onClick={() => setIsAuthModalOpen(false)}>&times;</button>
              <h2>{authMode === 'login' ? 'Welcome Back!' : 'Create Account'}</h2>
            </div>

            <div className="auth-body">
              <div className="auth-tabs">
                <button 
                  className={`auth-tab ${authMode === 'login' ? 'active' : ''}`} 
                  onClick={() => setAuthMode('login')}
                >Login</button>
                <button 
                  className={`auth-tab ${authMode === 'signup' ? 'active' : ''}`} 
                  onClick={() => setAuthMode('signup')}
                >Sign Up</button>
              </div>

              {authMode === 'login' ? (
                <form className="auth-form active" onSubmit={handleLogin}>
                  <div className="form-group">
                    <label>Email</label>
                    <div className="input-wrapper">
                      <i className="fas fa-envelope"></i>
                      <input 
                        type="email" 
                        placeholder="Enter email" 
                        required 
                        value={loginEmail}
                        onChange={e => setLoginEmail(e.target.value)}
                      />
                    </div>
                  </div>
                  <div className="form-group">
                    <label>Password</label>
                    <div className="input-wrapper">
                      <i className="fas fa-lock"></i>
                      <input 
                        type="password" 
                        placeholder="Enter password" 
                        required 
                        value={loginPass}
                        onChange={e => setLoginPass(e.target.value)}
                      />
                    </div>
                  </div>
                  <button type="submit" className="submit-btn">Sign In</button>
                </form>
              ) : (
                <form className="auth-form active" onSubmit={handleSignup}>
                  <div className="form-group">
                    <label>Name</label>
                    <div className="input-wrapper">
                      <i className="fas fa-user"></i>
                      <input 
                        type="text" 
                        placeholder="Name" 
                        required 
                        value={signupName}
                        onChange={e => setSignupName(e.target.value)}
                      />
                    </div>
                  </div>
                  <div className="form-group">
                    <label>Email</label>
                    <div className="input-wrapper">
                      <i className="fas fa-envelope"></i>
                      <input 
                        type="email" 
                        placeholder="Email" 
                        required 
                        value={signupEmail}
                        onChange={e => setSignupEmail(e.target.value)}
                      />
                    </div>
                  </div>
                  <div className="form-group">
                    <label>Password</label>
                    <div className="input-wrapper">
                      <i className="fas fa-lock"></i>
                      <input 
                        type="password" 
                        placeholder="Password" 
                        required 
                        value={signupPass}
                        onChange={e => setSignupPass(e.target.value)}
                      />
                    </div>
                  </div>
                  <button type="submit" className="submit-btn">Create Account</button>
                </form>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Cart Sidebar */}
      <div className={`cart-overlay ${isCartOpen ? 'show' : ''}`} onClick={() => setIsCartOpen(false)}></div>
      <aside className={`cart-sidebar ${isCartOpen ? 'open' : ''}`}>
        <div className="cart-header">
          <h3>Your Cart</h3>
          <button className="close-cart" onClick={() => setIsCartOpen(false)}>&times;</button>
        </div>
        <div className="cart-items">
          {cart.length === 0 ? (
            <div className="empty-cart"><p>Your cart is empty</p></div>
          ) : (
            cart.map((item, idx) => (
              <div key={idx} className="cart-item">
                <div className="cart-item-details">
                  <div className="cart-item-name">{item.name}</div>
                  <div className="cart-item-price">RM{item.price} x {item.quantity}</div>
                  <div className="remove-item" onClick={() => removeFromCart(item.name)}>Remove</div>
                </div>
              </div>
            ))
          )}
        </div>
        {cart.length > 0 && (
          <div className="cart-footer">
            <div className="cart-total">
              <span>Total:</span>
              <span>${cart.reduce((a, b) => a + (b.price * b.quantity), 0).toFixed(2)}</span>
            </div>
            <button className="checkout-btn">Checkout</button>
          </div>
        )}
      </aside>

      {/* Toast */}
      <div className={`toast ${showToast ? 'show' : ''} ${toastType}`}>
        <i className="fas fa-check-circle"></i>
        <span>{toastMsg}</span>
      </div>

    </div>
  );
}

export default App;
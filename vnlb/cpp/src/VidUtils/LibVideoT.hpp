/* Modified work: Copyright (c) 2019, Pablo Arias <pariasm@gmail.com>
 * Original work: Copyright (c) 2013, Marc Lebrun <marc.lebrun.ik@gmail.com>
 * 
 * This program is free software: you can use, modify and/or redistribute it
 * under the terms of the GNU Affero General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version. You should have received a copy of this license
 * along this program. If not, see <http://www.gnu.org/licenses/>.
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 */
#ifndef LIB_VIDEOT_HPP_INCLUDED
#define LIB_VIDEOT_HPP_INCLUDED

#include <vector>
#include <string>
#include <stdexcept>
#include <cassert>
#include <climits>
#include <cstdio>
#include <cmath>
#include <unistd.h>

#include "mt19937ar.h"

// Boundary condition type
enum VideoBC {BC_SYMM, BC_CLIP};

/* Structure containing size informations of a video.
 *
 * width    : width of the image;
 * height   : height of the image;
 * channels : number of channels in the image;
 * frames   : number of frames in the video;
 * wh       : equal to width * height. Provided for convenience;
 * whc      : equal to width * height * channels. Provided for convenience.
 * whcf     : equal to width * height * frames * channels. Provided for convenience.
 * whf      : equal to width * height * frames. Provided for convenience.
 */
struct VideoSize
{
	unsigned width;
	unsigned height;
	unsigned frames;
	unsigned channels;
	unsigned wh;
	unsigned whc;
	unsigned whcf;
	unsigned whf;

	// Constuctors
	VideoSize(void)
		: width(0), height(0), frames(0), channels(0)
	{
		update_fields();
	}

	VideoSize(unsigned w, unsigned h, unsigned f, unsigned c)
		: width(w), height(h), frames(f), channels(c)
	{
		update_fields();
	}

	// Comparison operators
	inline bool operator == (const VideoSize& sz) const
	{
		return (width    == sz.width     &&
		        height   == sz.height    &&
		        channels == sz.channels  &&
		        frames   == sz.frames    );
	}

	inline bool operator != (const VideoSize& sz) const
	{ 
		return !operator==(sz);
	}

	// Updates products of dimensions
	inline void update_fields(void)
	{
		wh = width * height;
		whc = wh * channels;
		whcf = whc * frames;
		whf  = wh  * frames;
	}

	// Returns index
	inline unsigned index(unsigned x, unsigned y, unsigned t, unsigned c) const
	{
		assert(x < width && y < height && t < frames && c < channels);
		return t*whc + c*wh + y*width + x;
	}

	inline unsigned index(VideoBC bc_type, unsigned x, unsigned y,
			      unsigned t, unsigned c) const
	{
		assert(c >= 0 && c < sz.channels);

		switch (bc_type)
		{
			case BC_SYMM:
				// NOTE: assumes that -width+1 < x < 2*(width -1)
				assert( -(int)width  < x && x < 2*(int)width  - 1 &&
				        -(int)height < y && y < 2*(int)height - 1 &&
				        -(int)frames < t && t < 2*(int)frames - 1 );

				x = (x < 0) ? -x : (x >= (int)width ) ? 2*(int)width  - 2 - x : x ;
				y = (y < 0) ? -y : (y >= (int)height) ? 2*(int)height - 2 - y : y ;
				t = (t < 0) ? -t : (t >= (int)frames) ? 2*(int)frames - 2 - t : t ;
				break;

			case BC_CLIP:
				x = (x < 0) ? 0 : (x >= (int)width ) ? (int)width  - 1 : x ;
				y = (y < 0) ? 0 : (y >= (int)height) ? (int)height - 1 : y ;
				t = (t < 0) ? 0 : (t >= (int)frames) ? (int)frames - 1 : t ;
				break;

			default:
				throw std::runtime_error("VideoSize::index(bc_type,x,y,t,c): unsupported bc_type");
		}

		return t*whc + c*wh + y*width + x;
	}

	// Returns index assuming the video has one channel
	inline unsigned index(unsigned x, unsigned y, unsigned t) const
	{
		assert(x < width && y < height && t < frames);
		return t*wh + y*width + x;
	}

	inline unsigned index(VideoBC bc_type, unsigned x, unsigned y, unsigned t) const
	{
		return index(bc_type, x, y, t, 0);
	}

	// Compute coordinates from index
	inline
	void coords(unsigned idx, unsigned& x, unsigned& y,
		    unsigned& t, unsigned& c) const
	{
		assert(idx < whcf);
		t = (idx      ) / whc;
		c = (idx % whc) / wh ;
		y = (idx % wh ) / width;
		x = (idx % width  );
	}

	// Compute coordinates from index assuming the video has one channel
	inline
	void coords(unsigned idx, unsigned& x, unsigned& y, unsigned& t) const
	{
		assert(idx < whf);
		t = (idx      ) / wh;
		y = (idx % wh ) / width;
		x = (idx % width  );
	}
};

//
// template class def
//

/* A video class template with very basic functionalities.
 *
 * NOTE: should not be used with T = bool, since current implementation
 * relies on std::vector, and std::vector<bool> cannot return a non-
 * constant reference to an element of the array.
 *
 * sz       : VideoSize structure with size of the video;
 * data     : pointer to an std::vector<T> containing the data
 */
template <class T>
class Video
{
	public:

		// Size
		VideoSize sz;

		// Data
		std::vector<T> data;

		// Constructors
		Video(void); // empty
		Video(const Video& in); // copy
		Video(const std::string pathToFiles,
		      unsigned firstFrame, unsigned lastFrame, unsigned frameStep = 1); // from filename
		Video(unsigned width, unsigned height, unsigned frames, unsigned channels = 1);  // alloc
		Video(unsigned width, unsigned height, unsigned frames, unsigned channels, T val);  // init
		Video(const VideoSize& size);  // alloc
		Video(const VideoSize& size, T val);  // init

		// Destructor
		~Video(void) { };

  
		void clear(void);
		void resize(unsigned width, unsigned height,
			    unsigned frames, unsigned channels = 1);
		void resize(const VideoSize& size);

		// Read/write pixel access ~ inline for efficiency
		T& operator () (unsigned idx); // from coordinates
		T& operator () (unsigned x, unsigned y, unsigned t, unsigned c = 0); // from coordinates
		T& operator () (VideoBC bc_type, int x, int y, int t, int c = 0); // with boundary conditions

		// Read only pixel access ~ inline for efficiency
		T operator () (unsigned idx) const; // from index
		T operator () (unsigned x, unsigned y, unsigned t, unsigned c = 0) const; // from coordinates
		T operator () (VideoBC bc_type, int x, int y, int t, int c = 0) const; // with boundary conditions

		// Pixel access with special boundary conditions
		T& getPixelSymmetric(int x, int y, int t, unsigned c = 0);
		T  getPixelSymmetric(int x, int y, int t, unsigned c = 0) const;
		
		// I/O
		void loadVideo(const std::string pathToFiles, 
		               unsigned firstFrame, unsigned lastFrame, unsigned frameStep = 1);
                void loadVideoFromPtr(const T* ptr, int w, int h, int c, int t);
                void saveVideoToPtr(T* ptr);
		               
		void saveVideo(const std::string pathToFiles, 
		               unsigned firstFrame, unsigned frameStep = 1) const;
		void saveVideoAscii(const std::string prefix, 
		                    unsigned firstFrame, unsigned frameStep = 1) const;
};


// template<> void Video<float>::resize(const VideoSize& size);
// template<> void Video<float>::resize(const VideoSize& size);
	
// extern template class
// Video<float>::Video(const Video& in);

// extern template class
// Video<float>::Video(const std::string pathToFiles,unsigned firstFrame,
// 		unsigned lastFrame,unsigned frameStep);
	
// extern template class
// Video<float>::Video(unsigned width,unsigned height,
// 		unsigned frames,unsigned channels);

// extern template class
// Video<float>::Video(unsigned width,unsigned height,unsigned frames,
// 		unsigned channels,T val);

// extern template class
// Video<float>::Video(const VideoSize& size, T val);

// extern template class
// Video<float>::Video(const VideoSize& size);

//
// inline functions 
//

template <class T>
inline T& Video<T>::getPixelSymmetric(int x, int y, int t, unsigned c)
{
	// NOTE: assumes that -width+1 < x < 2*(width -1)
	assert(-(int)sz.width   < x && x < 2*(int)sz.width -1&&
	       -(int)sz.height  < y && y < 2*(int)sz.height-1&&
	       -(int)sz.frames  < t && t < 2*(int)sz.frames-1);
	// symmetrize
	x = (x < 0) ? -x : (x >= (int)sz.width  ) ? 2*(int)sz.width  - 2 - x : x ;
	y = (y < 0) ? -y : (y >= (int)sz.height ) ? 2*(int)sz.height - 2 - y : y ;
	t = (t < 0) ? -t : (t >= (int)sz.frames ) ? 2*(int)sz.frames - 2 - t : t ;

	return data[sz.index(x,y,t,c)];
}

template <class T>
inline T Video<T>::getPixelSymmetric(int x, int y, int t, unsigned c) const
{
	// NOTE: assumes that -width+1 < x < 2*(width -1)
	assert(-(int)sz.width   < x && x < 2*(int)sz.width  - 1 &&
	       -(int)sz.height  < y && y < 2*(int)sz.height - 1 &&
	       -(int)sz.frames  < t && t < 2*(int)sz.frames - 1 );
	// symmetrize
	x = (x < 0) ? -x : (x >= (int)sz.width  ) ? 2*(int)sz.width  - 2 - x : x ;
	y = (y < 0) ? -y : (y >= (int)sz.height ) ? 2*(int)sz.height - 2 - y : y ;
	t = (t < 0) ? -t : (t >= (int)sz.frames ) ? 2*(int)sz.frames - 2 - t : t ;

	return data[sz.index(x,y,t,c)];
}


// interface 
namespace VideoUtils
{
  /* Structure to store the position of a rectangular tile. It also has data
   * to describe the position of the tile when it corresponds to a rectangular
   * tiling (potentially with an added border) of a video. The tiles in a
   * tiling do not overlap. The tile is contained within a crop. A crop is the
   * union of a tile and a border surrounding it. This border is necessary for
   * any boundary conditions that the processing applied to the tile might
   * require.
   *
   * In this structure we store
   * 1) coordinates of the crop with respect to the video coordinate system
   * 2) position of the tile in a tiling (i.e. 2nd tile from the left, 3rd from the top)
   * 3) coordinates of the tile with respect to the video coordinate system
   */
  struct TilePosition
  {
    int origin_x; // x coordinate of top-left-front corner of crop
    int origin_y; // y coordinate of top-left-front corner of crop
    int origin_t; // t coordinate of top-left-front corner of crop

    int ending_x; // x coordinate of bottom-right-back corner of crop
    int ending_y; // y coordinate of bottom-right-back corner of crop
    int ending_t; // t coordinate of bottom-right-back corner of crop

    VideoSize source_sz; // size of source video

    int tile_x;    // x index of tile (0 <= tile_x < ntiles_x)
    int tile_y;    // y index of tile (0 <= tile_y < ntiles_y)
    int tile_t;    // t index of tile (0 <= tile_t < ntiles_t)

    int ntiles_x;  // total number of tiles in x direction
    int ntiles_y;  // total number of tiles in y direction
    int ntiles_t;  // total number of tiles in t direction

    int tile_origin_x; // x coordinate of top-left-front corner of tile
    int tile_origin_y; // y coordinate of top-left-front corner of tile
    int tile_origin_t; // t coordinate of top-left-front corner of tile

    int tile_ending_x; // x coordinate of bottom-right-back corner of tile
    int tile_ending_y; // y coordinate of bottom-right-back corner of tile
    int tile_ending_t; // t coordinate of bottom-right-back corner of tile
  };


  template <class T>
  void addNoise(Video<T> const& vid, Video<T> &vidNoisy, const float sigma);

  template <class T>
  void computePSNR(Video<T> const& vid1, Video<T> const& vid2,
		   float &psnr, float &rmse);

  template <class T>
  void computePSNR(Video<T> const& vid1, Video<T> const& vid2,
		   std::vector<float> &psnr, std::vector<float> &rmse);

  template <class T>
  void crop(Video<T> const &vid1, Video<T> &vid2,
	    int origin_t = INT_MAX, int origin_x = INT_MAX, int origin_y = INT_MAX);

  template <class T>
  void crop(Video<T> const &vid1, Video<T> &vid2, const int * const origin);

  template <class T>
  void crop(Video<T> const &vid1, Video<T> &vid2, TilePosition const &tile);

  template <class T>
  void addBorder(Video<T> const& vid1, Video<T> &vid2,
		 const unsigned border, const bool forward);

  // template <class T>
  // void transformColorSpace(Video<T> &vid, const bool forward);
  void transformColorSpace(Video<float> &vid, const bool forward);
  void transformColorSpace(Video<char> &vid, const bool forward);

  void determineFactor(const unsigned n, unsigned &a, unsigned &b);

  template <class T>
  void subDivideTight(Video<T> const& vid,std::vector<Video<T> > &vidSub,
		      std::vector<TilePosition> &tiles,
		      const int border,const int ntiles);
  void subDivideTight(Video<float> const& vid,std::vector<Video<float> > &vidSub,
		      std::vector<TilePosition> &tiles,
		      const int border,const int ntiles);

  template <class T>
  void subDivide(Video<T> const& vid,std::vector<Video<T> > &vidSub,
		 std::vector<TilePosition> &tiles,
		 const unsigned border,const unsigned ntiles);

  template <class T>
  void subDivide(Video<T> const& vid,std::vector<Video<T> > &vidSub,
		 const unsigned border,const unsigned ntiles);

  template <class T>
  void subBuild(std::vector<Video<T> > const& vidSub,
		Video<T> &vid, const unsigned border);

  template <class T>
  void subBuildTight(std::vector<Video<T> > const& vidSub,
		     Video<T> &vid, const int border);
  void subBuildTight(std::vector<Video<float> > const& vidSub,
		     Video<float> &vid, const int border);

} // namespace VideoUtils


//
// explicit class decl
//

extern template class Video<float>;
extern template class Video<char>;


#endif // LIB_VIDEOT_HPP_INCLUDED

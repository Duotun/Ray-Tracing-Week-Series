#pragma once
#ifndef PDF_H
#define PDF_H

class hittable;
#include "vector.hpp"
#include "onb.hpp"
#include "utility.hpp"
#include "hittable.hpp"


class pdf {
public:
	virtual ~pdf(){}
	virtual double value(const Vector3& direction) const = 0;   //pdf value
	virtual Vector3 generate() const = 0;  //generated number related to pdf
};

class cosine_pdf : public pdf {
public:
	cosine_pdf(const Vector3& w) { uvw.build_from_w(w); }
	virtual double value(const Vector3& direction) const override {
		auto cosine = dot(unit_vector(direction), uvw.w());  //w component is the normal
		return (cosine <= 0) ? 0.0: cosine / pi;
	}
	virtual Vector3 generate() const override {
		return uvw.local(random_cosine_direction());
	}
	//onb member is ok
	onb uvw;

};

//toward a hittable like the light (emit simulate)
class hittable_pdf : public pdf {
public: 	
	hittable_pdf(shared_ptr<hittable> p, const point3& origin) : ptr(p), o(origin) {}

	virtual double value(const Vector3& direction) const override {
		return ptr->pdf_value(o, direction);
	}

	virtual Vector3 generate() const override {
		return ptr->random(o);
	}

public:
	point3 o;
	shared_ptr<hittable> ptr;
};


//linear mixture and use probability for vector generation
class mixture_pdf : public pdf {   //assume sctter pdf and direct ray pdf
public:
	mixture_pdf(shared_ptr<pdf>p0, shared_ptr<pdf>p1)
	{
		p[0] = p0;
		p[1] = p1;
	}
	//mix pdf value
	virtual double value(const Vector3& direction) const override {
		return 0.5 * p[0]->value(direction) + 0.5 * p[1]->value(direction);
	}

	virtual Vector3 generate() const override {
		if (random_double() < 0.5)
			return p[0]->generate();
		else
			return p[1]->generate();
	}
	//members, two pdf mixed
	shared_ptr<pdf>p[2];
};

// put here for the correct compiline sequence
//define a scatter record to help differentiate specular and diffuse material
struct scatter_record {
	ray specular_ray;
	bool is_specular;
	color attenuation;
	shared_ptr<pdf> pdf_ptr;
};


#endif // !pdf_h
